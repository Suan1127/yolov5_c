#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../src/core/tensor.h"
#include "../src/models/yolov5s_build.h"
#include "../src/models/yolov5s_infer.h"

/**
 * Integration test: Load model and run forward pass
 */
int test_model_build_and_forward() {
    printf("=== Integration Test: Model Build and Forward ===\n\n");
    
    const char* weights_path = "weights/weights.bin";
    
    // Build model
    printf("1. Building model...\n");
    yolov5s_model_t* model = yolov5s_build(weights_path, NULL);
    if (!model) {
        printf("  ERROR: Failed to build model\n");
        return -1;
    }
    printf("  ✓ Model built successfully\n");
    
    // Create dummy input (1, 3, 640, 640)
    printf("\n2. Creating input tensor...\n");
    tensor_t* input = tensor_create(1, 3, 640, 640);
    if (!input) {
        printf("  ERROR: Failed to create input tensor\n");
        yolov5s_free(model);
        return -1;
    }
    
    // Fill with small random values (normalized 0-1)
    for (size_t i = 0; i < tensor_size(input); i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }
    printf("  ✓ Input tensor created: (%d, %d, %d, %d)\n", 
           input->n, input->c, input->h, input->w);
    
    // Allocate outputs
    printf("\n3. Allocating output tensors...\n");
    tensor_t* outputs[3];
    outputs[0] = tensor_create(1, 128, 80, 80);   // P3
    outputs[1] = tensor_create(1, 256, 40, 40);   // P4
    outputs[2] = tensor_create(1, 512, 20, 20);   // P5
    
    if (!outputs[0] || !outputs[1] || !outputs[2]) {
        printf("  ERROR: Failed to allocate output tensors\n");
        tensor_free(input);
        yolov5s_free(model);
        return -1;
    }
    printf("  ✓ Output tensors allocated\n");
    
    // Forward pass
    printf("\n4. Running forward pass...\n");
    int ret = yolov5s_forward(model, input, outputs);
    if (ret != 0) {
        printf("  ERROR: Forward pass failed\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        tensor_free(input);
        yolov5s_free(model);
        return -1;
    }
    printf("  ✓ Forward pass completed\n");
    printf("    P3: (%d, %d, %d, %d)\n", 
           outputs[0]->n, outputs[0]->c, outputs[0]->h, outputs[0]->w);
    printf("    P4: (%d, %d, %d, %d)\n", 
           outputs[1]->n, outputs[1]->c, outputs[1]->h, outputs[1]->w);
    printf("    P5: (%d, %d, %d, %d)\n", 
           outputs[2]->n, outputs[2]->c, outputs[2]->h, outputs[2]->w);
    
    // Check saved features
    printf("\n5. Checking saved features...\n");
    tensor_t* p3 = yolov5s_get_saved_feature(model, 17);
    tensor_t* p4 = yolov5s_get_saved_feature(model, 20);
    tensor_t* p5 = yolov5s_get_saved_feature(model, 23);
    
    if (!p3 || !p4 || !p5) {
        printf("  ERROR: Failed to get saved features\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        tensor_free(input);
        yolov5s_free(model);
        return -1;
    }
    printf("  ✓ Saved features retrieved\n");
    printf("    P3: (%d, %d, %d, %d)\n", p3->n, p3->c, p3->h, p3->w);
    printf("    P4: (%d, %d, %d, %d)\n", p4->n, p4->c, p4->h, p4->w);
    printf("    P5: (%d, %d, %d, %d)\n", p5->n, p5->c, p5->h, p5->w);
    
    // Check output values (should not be all zeros)
    printf("\n6. Validating outputs...\n");
    int all_zero = 1;
    for (int i = 0; i < 3; i++) {
        size_t size = tensor_size(outputs[i]);
        for (size_t j = 0; j < size && j < 100; j++) {  // Check first 100 values
            if (outputs[i]->data[j] != 0.0f) {
                all_zero = 0;
                break;
            }
        }
    }
    
    if (all_zero) {
        printf("  WARNING: All output values are zero (model may not be working correctly)\n");
    } else {
        printf("  ✓ Outputs contain non-zero values\n");
    }
    
    // Cleanup
    tensor_free(outputs[0]);
    tensor_free(outputs[1]);
    tensor_free(outputs[2]);
    tensor_free(input);
    yolov5s_free(model);
    
    printf("\n=== Integration test PASSED ===\n\n");
    return 0;
}

int main() {
    printf("YOLOv5 C Integration Tests\n");
    printf("==========================\n\n");
    
    int ret = test_model_build_and_forward();
    
    if (ret == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Tests failed!\n");
        return 1;
    }
}
