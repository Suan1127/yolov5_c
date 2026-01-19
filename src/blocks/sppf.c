#include "sppf.h"
#include "../core/common.h"
#include <stdlib.h>
#include "../core/weights_loader.h"
#include "../ops/activation.h"
#include "../ops/concat.h"

int sppf_init(sppf_block_t* block, int32_t c1, int32_t c2, int32_t k) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(sppf_block_t));
    block->c1 = c1;
    block->c2 = c2;
    block->c_ = c1 / 2;  // hidden channels = c1 // 2
    
    // cv1: 1×1 conv, c1 -> c_
    conv2d_params_t cv1_params = {
        .out_channels = block->c_,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv1, c1, &cv1_params) != 0) return -1;
    
    batchnorm2d_params_t cv1_bn_params = {
        .num_features = block->c_,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv1_bn, block->c_, &cv1_bn_params) != 0) {
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // cv2: 1×1 conv, 4*c_ -> c2
    conv2d_params_t cv2_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv2, 4 * block->c_, &cv2_params) != 0) {
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv2_bn_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv2_bn, c2, &cv2_bn_params) != 0) {
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // MaxPool parameters: k×k, stride=1, padding=k//2
    block->pool_params.kernel_size = k;
    block->pool_params.stride = 1;
    block->pool_params.padding = k / 2;
    
    return 0;
}

void sppf_free(sppf_block_t* block) {
    if (block) {
        conv2d_free(&block->cv1);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv2_bn);
        memset(block, 0, sizeof(sppf_block_t));
    }
}

int sppf_forward(sppf_block_t* block, const tensor_t* input, tensor_t* output,
                 tensor_t* workspace1, tensor_t* workspace2, tensor_t* workspace3) {
    if (!block || !input || !output) return -1;
    
    // Allocate workspaces if not provided
    int need_free_ws1 = 0, need_free_ws2 = 0, need_free_ws3 = 0;
    
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) return -1;
        need_free_ws1 = 1;
    }
    
    if (!workspace2) {
        workspace2 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace2) {
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    
    if (!workspace3) {
        workspace3 = tensor_create(input->n, 4 * block->c_, input->h, input->w);
        if (!workspace3) {
            if (need_free_ws1) tensor_free(workspace1);
            if (need_free_ws2) tensor_free(workspace2);
            return -1;
        }
        need_free_ws3 = 1;
    }
    
    // cv1: c1 -> c_
    if (conv2d_forward(&block->cv1, input, workspace1) != 0) goto error;
    if (batchnorm2d_forward(&block->cv1_bn, workspace1, workspace1) != 0) goto error;
    activation_silu(workspace1);
    
    // y1 = x (workspace1)
    // y2 = m(y1) -> workspace2
    if (maxpool2d_forward(&block->pool_params, workspace1, workspace2) != 0) goto error;
    
    // Temporary: create y1, y2, y3, y4 tensors
    tensor_t* y1 = workspace1;  // Already computed (cv1 output)
    tensor_t* y2 = workspace2;   // Will be computed
    tensor_t* y3_temp = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* y4_temp = tensor_create(input->n, block->c_, input->h, input->w);
    
    if (!y3_temp || !y4_temp) {
        if (y3_temp) tensor_free(y3_temp);
        if (y4_temp) tensor_free(y4_temp);
        goto error;
    }
    
    // Copy y1 to keep it
    tensor_t* y1_copy = tensor_create(input->n, block->c_, input->h, input->w);
    if (!y1_copy) {
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    tensor_copy(y1_copy, y1);
    
    // y2 = m(y1)
    if (maxpool2d_forward(&block->pool_params, y1, y2) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    
    // y3 = m(y2)
    if (maxpool2d_forward(&block->pool_params, y2, y3_temp) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    
    // y4 = m(y3)
    if (maxpool2d_forward(&block->pool_params, y3_temp, y4_temp) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    
    // Concat: [y1_copy, y2, y3_temp, y4_temp] -> workspace3
    const tensor_t* concat_inputs[4] = {y1_copy, y2, y3_temp, y4_temp};
    if (concat_forward(concat_inputs, 4, workspace3) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    
    // cv2: 4*c_ -> c2
    if (conv2d_forward(&block->cv2, workspace3, output) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    if (batchnorm2d_forward(&block->cv2_bn, output, output) != 0) {
        tensor_free(y1_copy);
        tensor_free(y3_temp);
        tensor_free(y4_temp);
        goto error;
    }
    activation_silu(output);
    
    tensor_free(y1_copy);
    tensor_free(y3_temp);
    tensor_free(y4_temp);
    
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    if (need_free_ws3) tensor_free(workspace3);
    return 0;
    
error:
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    if (need_free_ws3) tensor_free(workspace3);
    return -1;
}

int sppf_load_weights(sppf_block_t* block, void* weights_loader, const char* prefix) {
    if (!block || !weights_loader) return -1;
    
    weights_loader_t* loader = (weights_loader_t*)weights_loader;
    char name[256];
    int32_t shape[4];
    int num_dims;
    
    // Load cv1
    snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) conv2d_load_weights(&block->cv1, w, NULL);
    
    snprintf(name, sizeof(name), "%s.cv1.bn.weight", prefix);
    float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.bias", prefix);
    float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.running_mean", prefix);
    float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.running_var", prefix);
    float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(&block->cv1_bn, bn_w, bn_b, bn_mean, bn_var);
    }
    
    // Load cv2
    snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) conv2d_load_weights(&block->cv2, w, NULL);
    
    snprintf(name, sizeof(name), "%s.cv2.bn.weight", prefix);
    bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.bias", prefix);
    bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.running_mean", prefix);
    bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.running_var", prefix);
    bn_var = weights_loader_get(loader, name, shape, &num_dims);
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(&block->cv2_bn, bn_w, bn_b, bn_mean, bn_var);
    }
    
    return 0;
}
