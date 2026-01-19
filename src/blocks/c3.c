#include "c3.h"
#include "../core/common.h"
#include <stdlib.h>
#include <stdio.h>
#include "../core/weights_loader.h"
#include "../ops/activation.h"
#include "../ops/concat.h"

int c3_init(c3_block_t* block, int32_t c1, int32_t c2, int32_t n, int shortcut) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(c3_block_t));
    block->c1 = c1;
    block->c2 = c2;
    block->c_ = c2 / 2;  // hidden channels = c2 * 0.5
    block->n = n;
    block->shortcut = shortcut;
    
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
    
    // cv2: 1×1 conv, c1 -> c_ (skip path)
    conv2d_params_t cv2_params = {
        .out_channels = block->c_,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv2, c1, &cv2_params) != 0) {
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv2_bn_params = {
        .num_features = block->c_,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv2_bn, block->c_, &cv2_bn_params) != 0) {
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // cv3: 1×1 conv, 2*c_ -> c2
    conv2d_params_t cv3_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv3, 2 * block->c_, &cv3_params) != 0) {
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv3_bn_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv3_bn, c2, &cv3_bn_params) != 0) {
        conv2d_free(&block->cv3);
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // Initialize bottlenecks
    if (n > 0) {
        block->bottlenecks = (bottleneck_t*)calloc(n, sizeof(bottleneck_t));
        if (!block->bottlenecks) {
            batchnorm2d_free(&block->cv3_bn);
            conv2d_free(&block->cv3);
            batchnorm2d_free(&block->cv2_bn);
            conv2d_free(&block->cv2);
            batchnorm2d_free(&block->cv1_bn);
            conv2d_free(&block->cv1);
            return -1;
        }
        
        for (int i = 0; i < n; i++) {
            if (bottleneck_init(&block->bottlenecks[i], block->c_, block->c_, shortcut) != 0) {
                // Cleanup
                for (int j = 0; j < i; j++) {
                    bottleneck_free(&block->bottlenecks[j]);
                }
                free(block->bottlenecks);
                batchnorm2d_free(&block->cv3_bn);
                conv2d_free(&block->cv3);
                batchnorm2d_free(&block->cv2_bn);
                conv2d_free(&block->cv2);
                batchnorm2d_free(&block->cv1_bn);
                conv2d_free(&block->cv1);
                return -1;
            }
        }
    }
    
    return 0;
}

void c3_free(c3_block_t* block) {
    if (block) {
        conv2d_free(&block->cv1);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv3);
        batchnorm2d_free(&block->cv3_bn);
        
        if (block->bottlenecks) {
            for (int i = 0; i < block->n; i++) {
                bottleneck_free(&block->bottlenecks[i]);
            }
            free(block->bottlenecks);
        }
        
        memset(block, 0, sizeof(c3_block_t));
    }
}

int c3_forward(c3_block_t* block, const tensor_t* input, tensor_t* output, 
               tensor_t* workspace1, tensor_t* workspace2) {
    if (!block || !input || !output) {
        fprintf(stderr, "Error: c3_forward: NULL pointer\n");
        return -1;
    }
    
    // Allocate workspaces if not provided
    int need_free_ws1 = 0, need_free_ws2 = 0;
    
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate workspace1 (%d, %d, %d, %d)\n",
                    input->n, block->c_, input->h, input->w);
            return -1;
        }
        need_free_ws1 = 1;
    }
    
    if (!workspace2) {
        workspace2 = tensor_create(input->n, 2 * block->c_, input->h, input->w);
        if (!workspace2) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate workspace2 (%d, %d, %d, %d)\n",
                    input->n, 2 * block->c_, input->h, input->w);
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    
    // Main path: cv1 -> bottlenecks -> (stored in workspace1)
    if (conv2d_forward(&block->cv1, input, workspace1) != 0) {
        fprintf(stderr, "Error: c3_forward: cv1 conv2d failed\n");
        goto error;
    }
    if (batchnorm2d_forward(&block->cv1_bn, workspace1, workspace1) != 0) {
        fprintf(stderr, "Error: c3_forward: cv1 batchnorm failed\n");
        goto error;
    }
    activation_silu(workspace1);
    
    // Pass through bottlenecks
    // Bottlenecks need temporary buffer (c_ channels)
    // For n > 1, we need separate buffer for intermediate bottlenecks
    tensor_t* bottleneck_temp = NULL;
    if (block->n > 1) {
        bottleneck_temp = tensor_create(input->n, block->c_, input->h, input->w);
        if (!bottleneck_temp) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate bottleneck_temp\n");
            goto error;
        }
    }
    
    tensor_t* bottleneck_input = workspace1;
    for (int i = 0; i < block->n; i++) {
        // For last bottleneck, output goes to workspace1 (reuse)
        // For others, use bottleneck_temp
        tensor_t* bottleneck_output = (i == block->n - 1) ? workspace1 : bottleneck_temp;
        if (bottleneck_forward(&block->bottlenecks[i], bottleneck_input, bottleneck_output, NULL) != 0) {
            fprintf(stderr, "Error: c3_forward: bottleneck %d failed\n", i);
            if (bottleneck_temp) tensor_free(bottleneck_temp);
            goto error;
        }
        bottleneck_input = bottleneck_output;
    }
    
    if (bottleneck_temp) tensor_free(bottleneck_temp);
    
    // Skip path: cv2
    // Need separate buffer for skip path output (c_ channels)
    tensor_t* skip_output = tensor_create(input->n, block->c_, input->h, input->w);
    if (!skip_output) {
        fprintf(stderr, "Error: c3_forward: Failed to allocate skip_output\n");
        goto error;
    }
    
    if (conv2d_forward(&block->cv2, input, skip_output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv2 conv2d failed\n");
        tensor_free(skip_output);
        goto error;
    }
    if (batchnorm2d_forward(&block->cv2_bn, skip_output, skip_output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv2 batchnorm failed\n");
        tensor_free(skip_output);
        goto error;
    }
    // Note: cv2 doesn't have activation in original YOLOv5
    
    // Concat: [workspace1 (main path), skip_output] -> workspace2
    const tensor_t* concat_inputs[2] = {workspace1, skip_output};
    if (concat_forward(concat_inputs, 2, workspace2) != 0) {
        fprintf(stderr, "Error: c3_forward: concat failed\n");
        tensor_free(skip_output);
        goto error;
    }
    
    tensor_free(skip_output);  // Free skip_output after concat
    
    // cv3: 2*c_ -> c2
    // Verify tensor sizes before conv2d
    if (workspace2->c != 2 * block->c_) {
        fprintf(stderr, "Error: c3_forward: workspace2 channels mismatch. Expected %d, got %d\n",
                2 * block->c_, workspace2->c);
        goto error;
    }
    if (output->c != block->c2) {
        fprintf(stderr, "Error: c3_forward: output channels mismatch. Expected %d, got %d\n",
                block->c2, output->c);
        goto error;
    }
    if (block->cv3.in_channels != 2 * block->c_) {
        fprintf(stderr, "Error: c3_forward: cv3 in_channels mismatch. Expected %d, got %d\n",
                2 * block->c_, block->cv3.in_channels);
        goto error;
    }
    
    if (conv2d_forward(&block->cv3, workspace2, output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv3 conv2d failed\n");
        fprintf(stderr, "  workspace2: (%d, %d, %d, %d)\n",
                workspace2->n, workspace2->c, workspace2->h, workspace2->w);
        fprintf(stderr, "  output: (%d, %d, %d, %d)\n",
                output->n, output->c, output->h, output->w);
        fprintf(stderr, "  cv3: in_channels=%d, out_channels=%d\n",
                block->cv3.in_channels, block->cv3.params.out_channels);
        goto error;
    }
    if (batchnorm2d_forward(&block->cv3_bn, output, output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv3 batchnorm failed\n");
        goto error;
    }
    activation_silu(output);
    
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    return 0;
    
error:
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    // Note: bottleneck_temp and skip_output are freed in their respective sections
    return -1;
}

int c3_load_weights(c3_block_t* block, void* weights_loader, const char* prefix) {
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
    
    // Load cv3
    snprintf(name, sizeof(name), "%s.cv3.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) conv2d_load_weights(&block->cv3, w, NULL);
    
    snprintf(name, sizeof(name), "%s.cv3.bn.weight", prefix);
    bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv3.bn.bias", prefix);
    bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv3.bn.running_mean", prefix);
    bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.cv3.bn.running_var", prefix);
    bn_var = weights_loader_get(loader, name, shape, &num_dims);
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(&block->cv3_bn, bn_w, bn_b, bn_mean, bn_var);
    }
    
    // Load bottlenecks
    for (int i = 0; i < block->n; i++) {
        snprintf(name, sizeof(name), "%s.m.%d", prefix, i);
        bottleneck_load_weights(&block->bottlenecks[i], loader, name);
    }
    
    return 0;
}
