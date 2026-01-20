#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "core/tensor.h"
#include "models/yolov5s_build.h"
#include "models/yolov5s_infer.h"
#include "postprocess/detect.h"
#include "postprocess/nms.h"

void print_usage(const char* prog_name) {
    printf("Usage: %s <image_name> [weights.bin] [model_meta.json]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  image_name        Image name (without extension, e.g., 'bus')\n");
    printf("                    Input tensor will be loaded from: data/inputs/<image_name>.bin\n");
    printf("  weights.bin       Model weights file (default: weights/weights.bin)\n");
    printf("  model_meta.json   Model metadata (default: weights/model_meta.json)\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s bus\n", prog_name);
    printf("  %s bus weights/weights.bin weights/model_meta.json\n", prog_name);
    printf("\n");
    printf("Note: Run 'python tools/preprocess.py' first to generate input tensors from images in data/images/\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* image_name = argv[1];
    // Default paths (will try multiple variations if not provided)
    const char* weights_path_arg = (argc >= 3) ? argv[2] : NULL;
    const char* model_meta_path_arg = (argc >= 4) ? argv[3] : NULL;
    
    // Construct input tensor path: data/inputs/<image_name>.bin
    char input_tensor_path[512];
    snprintf(input_tensor_path, sizeof(input_tensor_path), "data/inputs/%s.bin", image_name);
    
    printf("=== YOLOv5 C Inference ===\n\n");
    printf("Image name: %s\n", image_name);
    printf("\n");
    
    // Load input tensor
    printf("Loading input tensor...\n");
    
    // Try multiple path variations
    char paths[3][512];
    snprintf(paths[0], sizeof(paths[0]), "data/inputs/%s.bin", image_name);
    snprintf(paths[1], sizeof(paths[1]), "../data/inputs/%s.bin", image_name);
    snprintf(paths[2], sizeof(paths[2]), "../../data/inputs/%s.bin", image_name);
    
    const char* found_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(paths[i], "rb");
        if (test_fp) {
            fclose(test_fp);
            found_path = paths[i];
            break;
        }
    }
    
    if (!found_path) {
        fprintf(stderr, "Error: Cannot find input tensor file\n");
        fprintf(stderr, "Tried:\n");
        for (int i = 0; i < 3; i++) {
            fprintf(stderr, "  - %s\n", paths[i]);
        }
        #ifdef _WIN32
        char cwd[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd)) {
            fprintf(stderr, "Current directory: %s\n", cwd);
        }
        #else
        char* cwd = getcwd(NULL, 0);
        if (cwd) {
            fprintf(stderr, "Current directory: %s\n", cwd);
            free(cwd);
        }
        #endif
        fprintf(stderr, "\nHint: Run 'python tools/preprocess.py' to generate input tensors\n");
        fprintf(stderr, "Hint: Make sure you're running from build/Release or project root\n");
        return 1;
    }
    
    printf("Found input tensor: %s\n", found_path);
    tensor_t* input = tensor_load(found_path);
    if (!input) {
        fprintf(stderr, "Error: Failed to load input tensor from %s\n", found_path);
        fprintf(stderr, "Hint: The file exists but may be corrupted or in wrong format\n");
        return 1;
    }
    
    printf("Input shape: (%d, %d, %d, %d)\n", input->n, input->c, input->h, input->w);
    
    // Verify input shape (allow any size, but must be NCHW format)
    if (input->n != 1 || input->c != 3) {
        fprintf(stderr, "Error: Invalid input shape. Expected (1, 3, H, W), got (%d, %d, %d, %d)\n",
                input->n, input->c, input->h, input->w);
        tensor_free(input);
        return 1;
    }
    
    // Store input dimensions for later use
    int32_t input_h = input->h;
    int32_t input_w = input->w;
    int32_t input_size = input_h > input_w ? input_h : input_w;  // Use max dimension
    printf("Detected input size: %dx%d (using %d as reference)\n", input_h, input_w, input_size);
    
    // Find weights file
    printf("Finding weights file...\n");
    char weights_paths[3][512];
    if (weights_path_arg) {
        snprintf(weights_paths[0], sizeof(weights_paths[0]), "%s", weights_path_arg);
        snprintf(weights_paths[1], sizeof(weights_paths[1]), "../%s", weights_path_arg);
        snprintf(weights_paths[2], sizeof(weights_paths[2]), "../../%s", weights_path_arg);
    } else {
        snprintf(weights_paths[0], sizeof(weights_paths[0]), "weights/weights.bin");
        snprintf(weights_paths[1], sizeof(weights_paths[1]), "../weights/weights.bin");
        snprintf(weights_paths[2], sizeof(weights_paths[2]), "../../weights/weights.bin");
    }
    
    const char* found_weights_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(weights_paths[i], "rb");
        if (test_fp) {
            fclose(test_fp);
            found_weights_path = weights_paths[i];
            break;
        }
    }
    
    if (!found_weights_path) {
        fprintf(stderr, "Error: Cannot find weights file\n");
        fprintf(stderr, "Tried:\n");
        for (int i = 0; i < 3; i++) {
            fprintf(stderr, "  - %s\n", weights_paths[i]);
        }
        tensor_free(input);
        return 1;
    }
    printf("Found weights: %s\n", found_weights_path);
    
    // Find model meta file
    printf("Finding model meta file...\n");
    char meta_paths[3][512];
    if (model_meta_path_arg) {
        snprintf(meta_paths[0], sizeof(meta_paths[0]), "%s", model_meta_path_arg);
        snprintf(meta_paths[1], sizeof(meta_paths[1]), "../%s", model_meta_path_arg);
        snprintf(meta_paths[2], sizeof(meta_paths[2]), "../../%s", model_meta_path_arg);
    } else {
        snprintf(meta_paths[0], sizeof(meta_paths[0]), "weights/model_meta.json");
        snprintf(meta_paths[1], sizeof(meta_paths[1]), "../weights/model_meta.json");
        snprintf(meta_paths[2], sizeof(meta_paths[2]), "../../weights/model_meta.json");
    }
    
    const char* found_meta_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(meta_paths[i], "r");
        if (test_fp) {
            fclose(test_fp);
            found_meta_path = meta_paths[i];
            break;
        }
    }
    
    if (!found_meta_path) {
        printf("Warning: Model meta file not found, using defaults\n");
        printf("Tried:\n");
        for (int i = 0; i < 3; i++) {
            printf("  - %s\n", meta_paths[i]);
        }
    } else {
        printf("Found model meta: %s\n", found_meta_path);
    }
    printf("\n");
    
    // Build model
    printf("Building YOLOv5s model...\n");
    yolov5s_model_t* model = yolov5s_build(found_weights_path, found_meta_path);
    if (!model) {
        fprintf(stderr, "Error: Failed to build model\n");
        fprintf(stderr, "Check if weights file is valid: %s\n", found_weights_path);
        tensor_free(input);
        return 1;
    }
    printf("Model built successfully\n");
    
    // Allocate output tensors
    printf("\nAllocating output tensors...\n");
    tensor_t* outputs[3];
    outputs[0] = tensor_create(1, 128, 80, 80);   // P3
    outputs[1] = tensor_create(1, 256, 40, 40);   // P4
    outputs[2] = tensor_create(1, 512, 20, 20);   // P5
    
    if (!outputs[0] || !outputs[1] || !outputs[2]) {
        fprintf(stderr, "Error: Failed to allocate output tensors\n");
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    
    // Set output directory for saving intermediate layer tensors (for validation)
    // Only save intermediate tensors to testdata/c/, not final detection results
    // Final detection results are saved to data/outputs/ in the detection section below
    const char* output_dir = NULL;
    
    // Try multiple paths for testdata/c directory
    const char* test_paths[] = {
        "testdata/c",
        "../testdata/c",
        "../../testdata/c"
    };
    
    #ifdef _WIN32
    for (int i = 0; i < 3; i++) {
        char win_path[512];
        snprintf(win_path, sizeof(win_path), "%s", test_paths[i]);
        // Convert / to \ for Windows
        for (int j = 0; win_path[j]; j++) {
            if (win_path[j] == '/') win_path[j] = '\\';
        }
        DWORD attrs = GetFileAttributesA(win_path);
        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            output_dir = test_paths[i];
            printf("Found testdata/c directory: %s\n", output_dir);
            break;
        }
    }
    #else
    for (int i = 0; i < 3; i++) {
        if (access(test_paths[i], F_OK) == 0) {
            output_dir = test_paths[i];
            printf("Found testdata/c directory: %s\n", output_dir);
            break;
        }
    }
    #endif
    
    // Always try to set output directory (will create if doesn't exist)
    // Default to testdata/c
    if (!output_dir) {
        output_dir = "testdata/c";
        printf("Setting default output directory: %s (will be created if needed)\n", output_dir);
    }
    
    int set_dir_ret = yolov5s_set_output_dir(model, output_dir);
    if (set_dir_ret == 0) {
        printf("Intermediate layer output directory set to: %s\n", output_dir);
        printf("(Final detection results will be saved to data/outputs/)\n");
    } else {
        fprintf(stderr, "Warning: Failed to set output directory: %s\n", output_dir);
    }
    
    // Forward pass
    printf("\nRunning forward pass...\n");
    printf("This may take a while...\n");
    fflush(stdout);  // Ensure output is flushed
    
    int ret = yolov5s_forward(model, input, outputs);
    if (ret != 0) {
        fprintf(stderr, "\nError: Forward pass failed (return code: %d)\n", ret);
        fprintf(stderr, "Possible causes:\n");
        fprintf(stderr, "  - Memory allocation failure\n");
        fprintf(stderr, "  - Invalid layer configuration\n");
        fprintf(stderr, "  - Missing or corrupted weights\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Forward pass completed successfully\n");
    printf("P3 output: (%d, %d, %d, %d)\n", outputs[0]->n, outputs[0]->c, outputs[0]->h, outputs[0]->w);
    printf("P4 output: (%d, %d, %d, %d)\n", outputs[1]->n, outputs[1]->c, outputs[1]->h, outputs[1]->w);
    printf("P5 output: (%d, %d, %d, %d)\n", outputs[2]->n, outputs[2]->c, outputs[2]->h, outputs[2]->w);
    
    // Get P3, P4, P5 features for Detect head
    tensor_t* p3_feature = yolov5s_get_saved_feature(model, 17);
    tensor_t* p4_feature = yolov5s_get_saved_feature(model, 20);
    tensor_t* p5_feature = yolov5s_get_saved_feature(model, 23);
    
    if (!p3_feature || !p4_feature || !p5_feature) {
        fprintf(stderr, "Error: Failed to get detect features\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    
    // Detect head
    printf("\nRunning Detect head...\n");
    detect_params_t detect_params;
    // Use actual input size for detection (use max dimension as reference)
    // input_size is already declared above
    detect_init_params(&detect_params, 80, input_size);
    printf("Using input size: %d for detection (from input %dx%d)\n", input_size, input_h, input_w);
    
    detect_output_t detect_output;
    ret = detect_forward(p3_feature, p4_feature, p5_feature, &detect_output, &detect_params);
    if (ret != 0) {
        fprintf(stderr, "Error: Detect forward failed\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Detect head completed\n");
    
    // Decode detections
    printf("\nDecoding detections...\n");
    detection_t* detections = NULL;
    int32_t num_detections = 0;
    float conf_threshold = 0.25f;
    
    ret = detect_decode(&detect_output, &detect_params, &detections, &num_detections, conf_threshold);
    if (ret != 0) {
        fprintf(stderr, "Error: Decode failed\n");
        detect_free_output(&detect_output);
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Found %d detections (confidence > %.2f)\n", num_detections, conf_threshold);
    
    // NMS
    printf("\nRunning NMS...\n");
    detection_t* nms_detections = NULL;
    int32_t nms_count = 0;
    float iou_threshold = 0.45f;
    int32_t max_detections = 1000;
    
    ret = nms(detections, num_detections, &nms_detections, &nms_count, iou_threshold, max_detections);
    if (ret != 0) {
        fprintf(stderr, "Error: NMS failed\n");
        detect_free_detections(detections, num_detections);
        detect_free_output(&detect_output);
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5s_free(model);
        tensor_free(input);
        return 1;
    }
    printf("After NMS: %d detections\n", nms_count);
    
    // Print results
    printf("\n=== Detection Results ===\n");
    printf("Total detections: %d\n", nms_count);
    printf("\n");
    
    // Print top 10 detections
    int print_count = nms_count < 10 ? nms_count : 10;
    for (int i = 0; i < print_count; i++) {
        detection_t* det = &nms_detections[i];
        printf("Detection %d:\n", i + 1);
        printf("  Class ID: %d\n", det->cls_id);
        printf("  Confidence: %.4f\n", det->conf);
        printf("  BBox: (%.4f, %.4f, %.4f, %.4f)\n", det->x, det->y, det->w, det->h);
        printf("  Pixel coords: x=%.1f, y=%.1f, w=%.1f, h=%.1f\n",
               det->x * input_size, det->y * input_size, det->w * input_size, det->h * input_size);
        printf("\n");
    }
    
    // Save results to file
    printf("\nSaving results...\n");
    
    // Try multiple output paths
    char output_paths[3][512];
    snprintf(output_paths[0], sizeof(output_paths[0]), "data/outputs/%s_detections.txt", image_name);
    snprintf(output_paths[1], sizeof(output_paths[1]), "../data/outputs/%s_detections.txt", image_name);
    snprintf(output_paths[2], sizeof(output_paths[2]), "../../data/outputs/%s_detections.txt", image_name);
    
    FILE* fp = NULL;
    const char* saved_path = NULL;
    for (int i = 0; i < 3; i++) {
        fp = fopen(output_paths[i], "w");
        if (fp) {
            saved_path = output_paths[i];
            break;
        }
    }
    
    if (fp) {
        fprintf(fp, "Image: %s\n", image_name);
        fprintf(fp, "Total detections: %d\n", nms_count);
        fprintf(fp, "Format: class_id confidence x y w h (normalized 0-1)\n");
        fprintf(fp, "Format: class_id confidence x_pixel y_pixel w_pixel h_pixel\n");
        fprintf(fp, "\n");
        
        for (int i = 0; i < nms_count; i++) {
            detection_t* det = &nms_detections[i];
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f\n",
                    det->cls_id, det->conf,
                    det->x, det->y, det->w, det->h);
            fprintf(fp, "%d %.4f %.1f %.1f %.1f %.1f\n",
                    det->cls_id, det->conf,
                    det->x * input_size, det->y * input_size, det->w * input_size, det->h * input_size);
        }
        fclose(fp);
        printf("Results saved to: %s\n", saved_path);
    } else {
        fprintf(stderr, "Warning: Failed to save results\n");
        fprintf(stderr, "Tried paths:\n");
        for (int i = 0; i < 3; i++) {
            fprintf(stderr, "  - %s\n", output_paths[i]);
        }
    }
    
    // Cleanup
    detect_free_detections(nms_detections, nms_count);
    detect_free_detections(detections, num_detections);
    detect_free_output(&detect_output);
    tensor_free(outputs[0]);
    tensor_free(outputs[1]);
    tensor_free(outputs[2]);
    yolov5s_free(model);
    tensor_free(input);
    
    printf("\n=== Inference completed successfully ===\n");
    return 0;
}
