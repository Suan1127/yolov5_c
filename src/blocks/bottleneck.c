#include "bottleneck.h"
#include "../core/common.h"
#include <stdlib.h>
#include "../core/weights_loader.h"

int bottleneck_init(bottleneck_t* block, int32_t c1, int32_t c2, int shortcut) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(bottleneck_t));
    block->c1 = c1;
    block->c2 = c2;
    block->shortcut = shortcut;
    
    // Conv1: 1×1, c1 -> c2
    conv2d_params_t conv1_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->conv1, c1, &conv1_params) != 0) return -1;
    
    batchnorm2d_params_t bn1_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->bn1, c2, &bn1_params) != 0) {
        conv2d_free(&block->conv1);
        return -1;
    }
    
    // Conv2: 3×3, c2 -> c2
    conv2d_params_t conv2_params = {
        .out_channels = c2,
        .kernel_size = 3,
        .stride = 1,
        .padding = 1,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->conv2, c2, &conv2_params) != 0) {
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv1);
        return -1;
    }
    
    batchnorm2d_params_t bn2_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->bn2, c2, &bn2_params) != 0) {
        conv2d_free(&block->conv2);
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv1);
        return -1;
    }
    
    return 0;
}

void bottleneck_free(bottleneck_t* block) {
    if (block) {
        conv2d_free(&block->conv1);
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv2);
        batchnorm2d_free(&block->bn2);
        memset(block, 0, sizeof(bottleneck_t));
    }
}

int bottleneck_forward(bottleneck_t* block, const tensor_t* input, tensor_t* output, tensor_t* workspace) {
    if (!block || !input || !output) return -1;
    
    // workspace is used for intermediate results
    if (!workspace) {
        workspace = output;  // Use output as workspace if not provided
    }
    
    // Conv1 -> BN1 -> SiLU
    if (conv2d_forward(&block->conv1, input, workspace) != 0) return -1;
    if (batchnorm2d_forward(&block->bn1, workspace, workspace) != 0) return -1;
    activation_silu(workspace);
    
    // Conv2 -> BN2 -> SiLU
    tensor_t* temp = workspace;
    if (workspace == output) {
        // Need separate temp buffer
        temp = tensor_create(input->n, block->c2, input->h, input->w);
        if (!temp) return -1;
    }
    
    if (conv2d_forward(&block->conv2, workspace, temp) != 0) {
        if (temp != workspace) tensor_free(temp);
        return -1;
    }
    if (batchnorm2d_forward(&block->bn2, temp, temp) != 0) {
        if (temp != workspace) tensor_free(temp);
        return -1;
    }
    activation_silu(temp);
    
    // Add shortcut if enabled
    if (block->shortcut) {
        // output = input + temp
        for (size_t i = 0; i < tensor_size(input); i++) {
            output->data[i] = input->data[i] + temp->data[i];
        }
    } else {
        // output = temp
        tensor_copy(output, temp);
    }
    
    if (temp != workspace) {
        tensor_free(temp);
    }
    
    return 0;
}

int bottleneck_load_weights(bottleneck_t* block, void* weights_loader, const char* prefix) {
    if (!block || !weights_loader) return -1;
    
    weights_loader_t* loader = (weights_loader_t*)weights_loader;
    char name[256];
    
    // Load conv1 weights
    snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
    int32_t shape[4];
    int num_dims;
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) {
        conv2d_load_weights(&block->conv1, w, NULL);
    }
    
    // Load bn1 weights
    snprintf(name, sizeof(name), "%s.cv1.bn.weight", prefix);
    float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.bias", prefix);
    float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.running_mean", prefix);
    float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv1.bn.running_var", prefix);
    float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(&block->bn1, bn_w, bn_b, bn_mean, bn_var);
    }
    
    // Load conv2 weights
    snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) {
        conv2d_load_weights(&block->conv2, w, NULL);
    }
    
    // Load bn2 weights
    snprintf(name, sizeof(name), "%s.cv2.bn.weight", prefix);
    bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.bias", prefix);
    bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.running_mean", prefix);
    bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv2.bn.running_var", prefix);
    bn_var = weights_loader_get(loader, name, shape, &num_dims);
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(&block->bn2, bn_w, bn_b, bn_mean, bn_var);
    }
    
    return 0;
}
