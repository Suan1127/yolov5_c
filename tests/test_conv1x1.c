#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../src/core/tensor.h"
#include "../src/ops/conv2d.h"

int test_conv1x1_basic() {
    printf("Testing 1x1 convolution (basic)...\n");
    
    // Create input: (1, 3, 4, 4)
    tensor_t* input = tensor_create(1, 3, 4, 4);
    assert(input != NULL);
    
    // Fill with test data
    for (int i = 0; i < 3 * 4 * 4; i++) {
        input->data[i] = 1.0f;
    }
    
    // Create conv layer: 3 -> 8 channels, 1x1 kernel
    conv2d_params_t params = {
        .out_channels = 8,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    
    conv2d_layer_t layer;
    int ret = conv2d_init(&layer, 3, &params);
    assert(ret == 0);
    
    // Initialize weights: identity-like
    for (int oc = 0; oc < 8; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            layer.weight[oc * 3 + ic] = (oc == ic) ? 1.0f : 0.0f;
        }
    }
    
    // Initialize bias
    for (int oc = 0; oc < 8; oc++) {
        layer.bias[oc] = 0.0f;
    }
    
    // Create output
    tensor_t* output = tensor_create(1, 8, 4, 4);
    assert(output != NULL);
    
    // Forward pass
    ret = conv2d_forward(&layer, input, output);
    assert(ret == 0);
    
    // Verify output shape
    assert(output->n == 1);
    assert(output->c == 8);
    assert(output->h == 4);
    assert(output->w == 4);
    
    printf("  Output shape: (%d, %d, %d, %d)\n", output->n, output->c, output->h, output->w);
    printf("  First output value: %f\n", output->data[0]);
    
    // Cleanup
    conv2d_free(&layer);
    tensor_free(input);
    tensor_free(output);
    
    printf("  PASSED\n");
    return 0;
}

int main() {
    printf("=== Conv1x1 Tests ===\n\n");
    
    test_conv1x1_basic();
    
    printf("\n=== All tests passed ===\n");
    return 0;
}
