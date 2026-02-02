#include "yolov5n_infer.h"
#include "yolov5n_graph.h"
#include "../ops/activation.h"
#include "../ops/upsample.h"
#include "../ops/concat.h"
#include "../ops/conv2d.h"
#include "../core/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#endif

// Output directory for saving intermediate tensors
static char g_output_dir[512] = {0};

// Optional callback for per-layer stats (UART/verification)
static yolov5n_layer_stats_fn g_layer_stats_cb = NULL;

void yolov5n_set_layer_stats_callback(yolov5n_layer_stats_fn cb) {
    g_layer_stats_cb = cb;
}

// Timing utilities
#ifdef _WIN32
static double get_time_ms(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}
#else
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}
#endif

/* 레이어별 시간 구분: alloc(메모리 할당/해제), compute(연산), save(저장/복사) */
static void print_layer_timing(int layer_idx, double alloc_ms, double compute_ms, double save_ms) {
    double total = alloc_ms + compute_ms + save_ms;
    printf("    Layer %2d: alloc=%.2f ms  compute=%.2f ms  save=%.2f ms  total=%.2f ms\n",
           layer_idx, alloc_ms, compute_ms, save_ms, total);
    fflush(stdout);
}

// Forward declaration for conv2d_output_size helper
static void conv2d_output_size_helper(int32_t in_h, int32_t in_w, 
                                       int32_t kernel_size, int32_t stride, int32_t padding, int32_t dilation,
                                       int32_t* out_h, int32_t* out_w) {
    *out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    *out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// Helper: Save feature map
static void save_feature(yolov5n_model_t* model, int32_t layer_idx, tensor_t* feature) {
    // Map layer index to save array index (for FPN connections)
    // Save list: [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23]
    int32_t save_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23};
    int32_t save_idx = -1;
    
    for (int i = 0; i < 12; i++) {
        if (save_map[i] == layer_idx) {
            save_idx = i;
            break;
        }
    }
    
    // Save to saved_features array (for FPN connections) if in save_map
    if (save_idx >= 0) {
        // Free old feature if exists
        if (model->saved_features[save_idx]) {
            tensor_free(model->saved_features[save_idx]);
        }
        
        // Allocate and copy new feature
        model->saved_features[save_idx] = tensor_create(
            feature->n, feature->c, feature->h, feature->w
        );
        if (model->saved_features[save_idx]) {
            tensor_copy(model->saved_features[save_idx], feature);
        }
    }
    
    // Always save to file if output directory is set (for all layers 0-23)
    if (g_output_dir[0] != '\0' && feature && layer_idx >= 0 && layer_idx < 24) {
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/layer_%03d.bin", g_output_dir, layer_idx);
        int ret = tensor_dump(feature, filepath);
        if (ret != 0) {
            fprintf(stderr, "      Warning: Failed to save layer %d to %s\n", layer_idx, filepath);
        }
    }
    if (g_layer_stats_cb && feature && layer_idx >= 0 && layer_idx < 24) {
        g_layer_stats_cb(layer_idx, feature);
    }
}

// Helper: Get saved feature by layer index
tensor_t* yolov5n_get_saved_feature(yolov5n_model_t* model, int32_t layer_idx) {
    if (!model) return NULL;
    
    int32_t save_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23};
    for (int i = 0; i < 12; i++) {
        if (save_map[i] == layer_idx) {
            return model->saved_features[i];
        }
    }
    return NULL;
}

int yolov5n_get_detect_features(yolov5n_model_t* model, tensor_t** p3, tensor_t** p4, tensor_t** p5) {
    if (!model || !p3 || !p4 || !p5) return -1;
    
    *p3 = yolov5n_get_saved_feature(model, 17);  // P3 from layer 17
    *p4 = yolov5n_get_saved_feature(model, 20);  // P4 from layer 20
    *p5 = yolov5n_get_saved_feature(model, 23);  // P5 from layer 23
    
    if (!*p3 || !*p4 || !*p5) return -1;
    
    return 0;
}

int yolov5n_set_output_dir(yolov5n_model_t* model, const char* output_dir) {
    if (!model) return -1;
    
    if (output_dir == NULL || output_dir[0] == '\0') {
        g_output_dir[0] = '\0';
        return 0;
    }
    
    // Copy directory path
    strncpy(g_output_dir, output_dir, sizeof(g_output_dir) - 1);
    g_output_dir[sizeof(g_output_dir) - 1] = '\0';
    
    // Create directory if it doesn't exist
    #ifdef _WIN32
    // Try to create directory (ignore error if already exists)
    _mkdir(g_output_dir);
    #else
    mkdir(g_output_dir, 0755);
    #endif
    
    printf("  Output directory configured: %s\n", g_output_dir);
    fflush(stdout);
    
    return 0;
}

int yolov5n_save_features(yolov5n_model_t* model) {
    if (!model || g_output_dir[0] == '\0') return 0;
    
    int32_t save_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 20, 23};
    int saved_count = 0;
    
    for (int i = 0; i < 12; i++) {
        if (model->saved_features[i]) {
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/layer_%03d.bin", g_output_dir, save_map[i]);
            if (tensor_dump(model->saved_features[i], filepath) == 0) {
                saved_count++;
            }
        }
    }
    
    return (saved_count > 0) ? 0 : -1;
}

int yolov5n_forward(yolov5n_model_t* model, const tensor_t* input, tensor_t* output[3]) {
    if (!model || !input || !output) {
        fprintf(stderr, "Error: yolov5n_forward: NULL pointer\n");
        return -1;
    }
    
    // Start timing for entire forward pass (declared at function start for proper scope)
    double forward_start = get_time_ms();
    
    // Save input tensor if output directory is set
    if (g_output_dir[0] != '\0') {
        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/input.bin", g_output_dir);
        int ret = tensor_dump(input, filepath);
        if (ret == 0) {
            // Input tensor saved silently
            fflush(stdout);
        } else {
            fprintf(stderr, "  Warning: Failed to save input tensor to %s\n", filepath);
        }
    }
    
    // Verify input shape (allow any size, but must be NCHW format)
    if (input->n != 1 || input->c != 3) {
        fprintf(stderr, "Error: yolov5n_forward: Invalid input shape (%d, %d, %d, %d). Expected (1, 3, H, W)\n",
                input->n, input->c, input->h, input->w);
        return -1;
    }
    
    // Get actual input dimensions
    int32_t input_h = input->h;
    int32_t input_w = input->w;
    
    // Calculate output sizes dynamically based on input size
    // Helper function for conv output size (matches conv2d_output_size formula)
    // Formula: (in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
    // For dilation=1: (in + 2*padding - (kernel_size-1) - 1) / stride + 1 = (in + 2*padding - kernel_size) / stride + 1
    #define CONV_OUT(in, k, s, p, d) (((in) + 2 * (p) - (d) * ((k) - 1) - 1) / (s) + 1)
    // For dilation=1 (most common case):
    #define CONV_OUT_D1(in, k, s, p) (((in) + 2 * (p) - (k)) / (s) + 1)
    
    // Layer 0 output: Conv(6x6, s=2, p=2)
    int32_t l0_h = CONV_OUT_D1(input_h, 6, 2, 2);
    int32_t l0_w = CONV_OUT_D1(input_w, 6, 2, 2);
    printf("  Calculated Layer 0 output size: %dx%d\n", l0_h, l0_w);
    
    // Allocate workspace buffers (ping-pong)
    tensor_t* buf_a = NULL;
    tensor_t* buf_b = NULL;
    // Workspace buffers will be allocated on-demand by C3/SPPF blocks
    
    // ========== Backbone (0-9) ==========
    
    printf("  Backbone: Layers 0-9...\n");
    printf("  Input size: %dx%d\n", input_h, input_w);
    printf("  (alloc=memory alloc/free, compute=ops, save=copy)\n");
    fflush(stdout);
    
    // Layer 0: Conv(3->16, 6x6, s=2, p=2) for YOLOv5n
    printf("    Layer 0: Conv(3->%d, 6x6, s=2, p=2)...\n", model->backbone_convs[0].out_channels);
    fflush(stdout);
    double t0_0 = get_time_ms();
    buf_a = tensor_create(1, model->backbone_convs[0].out_channels, l0_h, l0_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 0\n");
        return -1;
    }
    double t0_alloc = get_time_ms();
    // Verify output size before forward pass
    int32_t expected_h = l0_h;
    int32_t expected_w = l0_w;
    if (buf_a->h != expected_h || buf_a->w != expected_w) {
        fprintf(stderr, "Error: Layer 0 output size mismatch. Expected (%d, %d), got (%d, %d)\n",
                expected_h, expected_w, buf_a->h, buf_a->w);
        goto error;
    }
    /* Fused Conv+BN+SiLU: Conv 결과를 DDR에 쓰지 않고 한 번만 기록 (메모리 최적화) */
    if (model->backbone_convs[0].is_fused) {
        printf("    Layer 0: BN skipped (fused)\n");
        fflush(stdout);
        if (conv2d_fused_bn_silu_forward(&model->backbone_convs[0].conv, NULL, input, buf_a) != 0) {
            fprintf(stderr, "Error: Conv2D fused forward failed at Layer 0\n");
            goto error;
        }
    } else {
        if (conv2d_fused_bn_silu_forward(&model->backbone_convs[0].conv, &model->backbone_convs[0].bn, input, buf_a) != 0) {
            fprintf(stderr, "Error: Conv2D fused forward failed at Layer 0\n");
            goto error;
        }
    }
    double t0_compute = get_time_ms();
    save_feature(model, 0, buf_a);
    double t0_save = get_time_ms();
    print_layer_timing(0, t0_alloc - t0_0, t0_compute - t0_alloc, t0_save - t0_compute);

    // Layer 1: Conv(16->32, 3x3, s=2) for YOLOv5n
    printf("    Layer 1: Conv(%d->%d, 3x3, s=2)...\n", model->backbone_convs[1].in_channels, model->backbone_convs[1].out_channels);
    fflush(stdout);
    int32_t l1_h = CONV_OUT_D1(l0_h, 3, 2, 1);
    int32_t l1_w = CONV_OUT_D1(l0_w, 3, 2, 1);
    double t1_0 = get_time_ms();
    buf_b = tensor_create(1, model->backbone_convs[1].out_channels, l1_h, l1_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 1\n");
        goto error;
    }
    double t1_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->backbone_convs[1].conv,
            model->backbone_convs[1].is_fused ? NULL : &model->backbone_convs[1].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 1\n");
        goto error;
    }
    double t1_compute = get_time_ms();
    save_feature(model, 1, buf_b);
    double t1_save = get_time_ms();
    print_layer_timing(1, t1_alloc - t1_0, t1_compute - t1_alloc, t1_save - t1_compute);
    
    // Swap buffers
    tensor_t* temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 2: C3(64->64, n=1)
    printf("    Layer 2: C3(64->64, n=1)...\n");
    fflush(stdout);
    double t2_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_c3s[0].c2, l1_h, l1_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate output buffer for Layer 2\n");
        goto error;
    }
    double t2_alloc = get_time_ms();
    if (c3_forward(&model->backbone_c3s[0].block, buf_a, buf_b, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 2\n");
        goto error;
    }
    double t2_compute = get_time_ms();
    save_feature(model, 2, buf_b);
    double t2_save = get_time_ms();
    print_layer_timing(2, t2_alloc - t2_0, t2_compute - t2_alloc, t2_save - t2_compute);
    
    // Swap: buf_a = old input (64), buf_b = C3 output (64)
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 3: Conv(32->64, 3x3, s=2) -> SAVE[3] for YOLOv5n
    printf("    Layer 3: Conv(%d->%d, 3x3, s=2)...\n", model->backbone_convs[2].in_channels, model->backbone_convs[2].out_channels);
    fflush(stdout);
    int32_t l3_h = CONV_OUT_D1(l1_h, 3, 2, 1);
    int32_t l3_w = CONV_OUT_D1(l1_w, 3, 2, 1);
    double t3_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_convs[2].out_channels, l3_h, l3_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 3\n");
        goto error;
    }
    double t3_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->backbone_convs[2].conv,
            model->backbone_convs[2].is_fused ? NULL : &model->backbone_convs[2].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 3\n");
        goto error;
    }
    double t3_compute = get_time_ms();
    save_feature(model, 3, buf_b);
    double t3_save = get_time_ms();
    print_layer_timing(3, t3_alloc - t3_0, t3_compute - t3_alloc, t3_save - t3_compute);
    
    // Swap: buf_a = old input (64), buf_b = Layer 3 output (128)
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 4: C3(64->64, n=2) -> SAVE[4] for YOLOv5n
    printf("    Layer 4: C3(%d->%d, n=2)...\n", model->backbone_c3s[1].c1, model->backbone_c3s[1].c2);
    fflush(stdout);
    double t4_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_c3s[1].c2, l3_h, l3_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate output buffer for Layer 4\n");
        goto error;
    }
    double t4_alloc = get_time_ms();
    if (c3_forward(&model->backbone_c3s[1].block, buf_a, buf_b, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 4\n");
        goto error;
    }
    double t4_compute = get_time_ms();
    save_feature(model, 4, buf_b);
    double t4_save = get_time_ms();
    print_layer_timing(4, t4_alloc - t4_0, t4_compute - t4_alloc, t4_save - t4_compute);
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 5: Conv(64->128, 3x3, s=2) -> SAVE[5] for YOLOv5n
    printf("    Layer 5: Conv(%d->%d, 3x3, s=2)...\n", model->backbone_convs[3].in_channels, model->backbone_convs[3].out_channels);
    fflush(stdout);
    int32_t l5_h = CONV_OUT_D1(l3_h, 3, 2, 1);
    int32_t l5_w = CONV_OUT_D1(l3_w, 3, 2, 1);
    double t5_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_convs[3].out_channels, l5_h, l5_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 5\n");
        goto error;
    }
    double t5_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->backbone_convs[3].conv,
            model->backbone_convs[3].is_fused ? NULL : &model->backbone_convs[3].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 5\n");
        goto error;
    }
    double t5_compute = get_time_ms();
    save_feature(model, 5, buf_b);
    double t5_save = get_time_ms();
    print_layer_timing(5, t5_alloc - t5_0, t5_compute - t5_alloc, t5_save - t5_compute);
    
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 6: C3(256->256, n=3) -> SAVE[6]
    printf("    Layer 6: C3(%d->%d, n=3)...\n", model->backbone_c3s[2].c1, model->backbone_c3s[2].c2);
    fflush(stdout);
    double t6_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_c3s[2].c2, l5_h, l5_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 6\n");
        goto error;
    }
    double t6_alloc = get_time_ms();
    if (c3_forward(&model->backbone_c3s[2].block, buf_a, buf_b, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 6\n");
        goto error;
    }
    double t6_compute = get_time_ms();
    save_feature(model, 6, buf_b);
    double t6_save = get_time_ms();
    print_layer_timing(6, t6_alloc - t6_0, t6_compute - t6_alloc, t6_save - t6_compute);
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 7: Conv(128->256, 3x3, s=2) -> SAVE[7] for YOLOv5n
    printf("    Layer 7: Conv(%d->%d, 3x3, s=2)...\n", model->backbone_convs[4].in_channels, model->backbone_convs[4].out_channels);
    fflush(stdout);
    int32_t l7_h = CONV_OUT_D1(l5_h, 3, 2, 1);
    int32_t l7_w = CONV_OUT_D1(l5_w, 3, 2, 1);
    double t7_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_convs[4].out_channels, l7_h, l7_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 7\n");
        goto error;
    }
    double t7_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->backbone_convs[4].conv,
            model->backbone_convs[4].is_fused ? NULL : &model->backbone_convs[4].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 7\n");
        goto error;
    }
    double t7_compute = get_time_ms();
    save_feature(model, 7, buf_b);
    double t7_save = get_time_ms();
    print_layer_timing(7, t7_alloc - t7_0, t7_compute - t7_alloc, t7_save - t7_compute);
    
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 8: C3(256->256, n=1) for YOLOv5n
    printf("    Layer 8: C3(%d->%d, n=1)...\n", model->backbone_c3s[3].c1, model->backbone_c3s[3].c2);
    fflush(stdout);
    double t8_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->backbone_c3s[3].c2, l7_h, l7_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 8\n");
        goto error;
    }
    double t8_alloc = get_time_ms();
    if (c3_forward(&model->backbone_c3s[3].block, buf_a, buf_b, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 8\n");
        goto error;
    }
    double t8_compute = get_time_ms();
    save_feature(model, 8, buf_b);
    double t8_save = get_time_ms();
    print_layer_timing(8, t8_alloc - t8_0, t8_compute - t8_alloc, t8_save - t8_compute);
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // Layer 9: SPPF(512→512, k=5) → SAVE[9] (buf_b 재사용, alloc 없음)
    printf("    Layer 9: SPPF(512->512, k=5)...\n");
    fflush(stdout);
    double t9_0 = get_time_ms();
    double t9_alloc = t9_0;
    if (sppf_forward(&model->sppf, buf_a, buf_b, NULL, NULL, NULL) != 0) {
        fprintf(stderr, "Error: SPPF forward failed at Layer 9\n");
        goto error;
    }
    double t9_compute = get_time_ms();
    save_feature(model, 9, buf_b);
    double t9_save = get_time_ms();
    print_layer_timing(9, t9_alloc - t9_0, t9_compute - t9_alloc, t9_save - t9_compute);
    temp = buf_a;
    buf_a = buf_b;
    buf_b = temp;
    
    // ========== Neck (10-23) ==========
    printf("  Backbone completed\n");
    printf("  Neck: Layers 10-23...\n");
    printf("  (alloc=memory alloc/free, compute=ops, save=copy)\n");
    fflush(stdout);
    
    // Layer 10: Conv(256->128, 1x1) for YOLOv5n
    printf("    Layer 10: Conv(%d->%d, 1x1)...\n", model->head_convs[0].in_channels, model->head_convs[0].out_channels);
    fflush(stdout);
    double t10_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->head_convs[0].out_channels, l7_h, l7_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 10\n");
        goto error;
    }
    double t10_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->head_convs[0].conv,
            model->head_convs[0].is_fused ? NULL : &model->head_convs[0].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 10\n");
        goto error;
    }
    double t10_compute = get_time_ms();
    save_feature(model, 10, buf_b);
    double t10_save = get_time_ms();
    print_layer_timing(10, t10_alloc - t10_0, t10_compute - t10_alloc, t10_save - t10_compute);
    
    // Save layer 10 output for later concat
    tensor_t* layer10_output = tensor_create(1, model->head_convs[0].out_channels, l7_h, l7_w);
    if (!layer10_output) goto error;
    tensor_copy(layer10_output, buf_b);
    
    // Layer 11: Upsample(x2) - double the size
    printf("    Layer 11: Upsample(x2)...\n");
    fflush(stdout);
    int32_t l11_h = l7_h * 2;
    int32_t l11_w = l7_w * 2;
    double t11_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_convs[0].out_channels, l11_h, l11_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 11\n");
        goto error;
    }
    double t11_alloc = get_time_ms();
    upsample_params_t upsample_params = {
        .scale_factor = 2,
        .mode = "nearest"
    };
    if (upsample_forward(&upsample_params, buf_b, buf_a) != 0) {
        fprintf(stderr, "Error: Upsample forward failed at Layer 11\n");
        goto error;
    }
    double t11_compute = get_time_ms();
    save_feature(model, 11, buf_a);
    double t11_save = get_time_ms();
    print_layer_timing(11, t11_alloc - t11_0, t11_compute - t11_alloc, t11_save - t11_compute);
    
    // Layer 12: Concat([11, 6])
    printf("    Layer 12: Concat([11, 6])...\n");
    fflush(stdout);
    tensor_t* layer6_feature = yolov5n_get_saved_feature(model, 6);
    if (!layer6_feature) {
        fprintf(stderr, "Error: Failed to get Layer 6 feature\n");
        goto error;
    }
    
    // Verify layer6_feature size matches
    if (layer6_feature->h != l11_h || layer6_feature->w != l11_w) {
        fprintf(stderr, "Error: Layer 6 feature size mismatch. Expected (%d, %d), got (%d, %d)\n",
                l11_h, l11_w, layer6_feature->h, layer6_feature->w);
        goto error;
    }
    
    double t12_0 = get_time_ms();
    tensor_free(buf_b);
    int32_t concat_12_channels = model->head_convs[0].out_channels + model->backbone_c3s[2].c2;  // Layer 11 + Layer 6
    buf_b = tensor_create(1, concat_12_channels, l11_h, l11_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 12\n");
        goto error;
    }
    double t12_alloc = get_time_ms();
    const tensor_t* concat_inputs[2] = {buf_a, layer6_feature};
    if (concat_forward(concat_inputs, 2, buf_b) != 0) {
        fprintf(stderr, "Error: Concat forward failed at Layer 12\n");
        goto error;
    }
    double t12_compute = get_time_ms();
    save_feature(model, 12, buf_b);
    double t12_save = get_time_ms();
    print_layer_timing(12, t12_alloc - t12_0, t12_compute - t12_alloc, t12_save - t12_compute);
    
    // Layer 13: C3(256->128, n=1, shortcut=False) for YOLOv5n
    printf("    Layer 13: C3(%d->%d, n=1, shortcut=False)...\n", model->head_c3s[0].c1, model->head_c3s[0].c2);
    fflush(stdout);
    double t13_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_c3s[0].c2, l11_h, l11_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 13\n");
        goto error;
    }
    double t13_alloc = get_time_ms();
    if (c3_forward(&model->head_c3s[0].block, buf_b, buf_a, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 13\n");
        goto error;
    }
    double t13_compute = get_time_ms();
    save_feature(model, 13, buf_a);
    double t13_save = get_time_ms();
    print_layer_timing(13, t13_alloc - t13_0, t13_compute - t13_alloc, t13_save - t13_compute);
    
    // Save layer 13 output for later concat
    tensor_t* layer13_output = tensor_create(1, model->head_c3s[0].c2, l11_h, l11_w);
    if (!layer13_output) goto error;
    tensor_copy(layer13_output, buf_a);
    
    // Layer 14: Conv(128->64, 1x1) for YOLOv5n
    printf("    Layer 14: Conv(%d->%d, 1x1)...\n", model->head_convs[1].in_channels, model->head_convs[1].out_channels);
    fflush(stdout);
    double t14_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->head_convs[1].out_channels, l11_h, l11_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 14\n");
        goto error;
    }
    double t14_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->head_convs[1].conv,
            model->head_convs[1].is_fused ? NULL : &model->head_convs[1].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 14\n");
        goto error;
    }
    double t14_compute = get_time_ms();
    save_feature(model, 14, buf_b);
    double t14_save = get_time_ms();
    print_layer_timing(14, t14_alloc - t14_0, t14_compute - t14_alloc, t14_save - t14_compute);
    
    // Save layer 14 output for later concat (Layer 19)
    tensor_t* layer14_output = tensor_create(1, model->head_convs[1].out_channels, l11_h, l11_w);
    if (!layer14_output) goto error;
    tensor_copy(layer14_output, buf_b);
    
    // Layer 15: Upsample(x2) - double the size
    printf("    Layer 15: Upsample(x2)...\n");
    fflush(stdout);
    int32_t l15_h = l11_h * 2;
    int32_t l15_w = l11_w * 2;
    double t15_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_convs[1].out_channels, l15_h, l15_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 15\n");
        goto error;
    }
    double t15_alloc = get_time_ms();
    if (upsample_forward(&upsample_params, buf_b, buf_a) != 0) {
        fprintf(stderr, "Error: Upsample forward failed at Layer 15\n");
        fprintf(stderr, "  Input: (%d, %d, %d, %d)\n", buf_b->n, buf_b->c, buf_b->h, buf_b->w);
        fprintf(stderr, "  Output: (%d, %d, %d, %d)\n", buf_a->n, buf_a->c, buf_a->h, buf_a->w);
        goto error;
    }
    double t15_compute = get_time_ms();
    if (g_layer_stats_cb) g_layer_stats_cb(15, buf_a);
    double t15_save = get_time_ms();
    print_layer_timing(15, t15_alloc - t15_0, t15_compute - t15_alloc, t15_save - t15_compute);
    
    // Layer 16: Concat([15, 4])
    printf("    Layer 16: Concat([15, 4])...\n");
    fflush(stdout);
    tensor_t* layer4_feature = yolov5n_get_saved_feature(model, 4);
    if (!layer4_feature) {
        fprintf(stderr, "Error: Failed to get Layer 4 feature\n");
        goto error;
    }
    
    // Verify layer4_feature size matches
    if (layer4_feature->h != l15_h || layer4_feature->w != l15_w) {
        fprintf(stderr, "Error: Layer 4 feature size mismatch. Expected (%d, %d), got (%d, %d)\n",
                l15_h, l15_w, layer4_feature->h, layer4_feature->w);
        goto error;
    }
    
    double t16_0 = get_time_ms();
    tensor_free(buf_b);
    int32_t concat_16_channels = model->head_convs[1].out_channels + model->backbone_c3s[1].c2;  // Layer 15 + Layer 4
    buf_b = tensor_create(1, concat_16_channels, l15_h, l15_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 16\n");
        goto error;
    }
    double t16_alloc = get_time_ms();
    const tensor_t* concat_inputs2[2] = {buf_a, layer4_feature};
    if (concat_forward(concat_inputs2, 2, buf_b) != 0) {
        fprintf(stderr, "Error: Concat forward failed at Layer 16\n");
        goto error;
    }
    double t16_compute = get_time_ms();
    save_feature(model, 16, buf_b);
    double t16_save = get_time_ms();
    print_layer_timing(16, t16_alloc - t16_0, t16_compute - t16_alloc, t16_save - t16_compute);
    
    // Layer 17: C3(128->64, n=1, shortcut=False) -> SAVE[17] (P3) for YOLOv5n
    printf("    Layer 17: C3(%d->%d, n=1, shortcut=False)...\n", model->head_c3s[1].c1, model->head_c3s[1].c2);
    fflush(stdout);
    double t17_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_c3s[1].c2, l15_h, l15_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 17\n");
        goto error;
    }
    double t17_alloc = get_time_ms();
    if (c3_forward(&model->head_c3s[1].block, buf_b, buf_a, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 17\n");
        goto error;
    }
    double t17_compute = get_time_ms();
    save_feature(model, 17, buf_a);
    double t17_save = get_time_ms();
    print_layer_timing(17, t17_alloc - t17_0, t17_compute - t17_alloc, t17_save - t17_compute);
    
    // Output[0] = P3 - resize if needed
    if (output[0]) {
        if (output[0]->h != l15_h || output[0]->w != l15_w) {
            tensor_free(output[0]);
            output[0] = tensor_create(1, model->head_c3s[1].c2, l15_h, l15_w);
            if (!output[0]) goto error;
        }
        tensor_copy(output[0], buf_a);
    }
    
    // Layer 18: Conv(64->64, 3x3, s=2) for YOLOv5n
    printf("    Layer 18: Conv(%d->%d, 3x3, s=2)...\n", model->head_convs[2].in_channels, model->head_convs[2].out_channels);
    fflush(stdout);
    int32_t l18_h = CONV_OUT_D1(l15_h, 3, 2, 1);
    int32_t l18_w = CONV_OUT_D1(l15_w, 3, 2, 1);
    double t18_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->head_convs[2].out_channels, l18_h, l18_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 18\n");
        goto error;
    }
    double t18_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->head_convs[2].conv,
            model->head_convs[2].is_fused ? NULL : &model->head_convs[2].bn, buf_a, buf_b) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 18\n");
        goto error;
    }
    double t18_compute = get_time_ms();
    save_feature(model, 18, buf_b);
    double t18_save = get_time_ms();
    print_layer_timing(18, t18_alloc - t18_0, t18_compute - t18_alloc, t18_save - t18_compute);
    
    // Layer 19: Concat([18, 14]) - according to YAML: [[-1, 14], 1, Concat, [1]]
    printf("    Layer 19: Concat([18, 14])...\n");
    fflush(stdout);
    
    // If layer14_output size doesn't match, we need to resize it
    tensor_t* layer14_resized = layer14_output;
    if (layer14_output->h != l18_h || layer14_output->w != l18_w) {
        printf("      Resizing Layer 14 output from (%d, %d) to (%d, %d)\n", 
               layer14_output->h, layer14_output->w, l18_h, l18_w);
        fflush(stdout);
        
        // Create resized tensor
        layer14_resized = tensor_create(1, layer14_output->c, l18_h, l18_w);
        if (!layer14_resized) {
            fprintf(stderr, "Error: Failed to allocate resized Layer 14 output\n");
            goto error;
        }
        
        // Simple nearest-neighbor downsampling (since we're going from larger to smaller)
        float scale_h = (float)layer14_output->h / l18_h;
        float scale_w = (float)layer14_output->w / l18_w;
        
        for (int32_t b = 0; b < layer14_resized->n; b++) {
            for (int32_t c = 0; c < layer14_resized->c; c++) {
                for (int32_t h = 0; h < l18_h; h++) {
                    for (int32_t w = 0; w < l18_w; w++) {
                        int32_t src_h = (int32_t)(h * scale_h);
                        int32_t src_w = (int32_t)(w * scale_w);
                        if (src_h >= layer14_output->h) src_h = layer14_output->h - 1;
                        if (src_w >= layer14_output->w) src_w = layer14_output->w - 1;
                        
                        const float* src_val = tensor_at_const(layer14_output, b, c, src_h, src_w);
                        *tensor_at(layer14_resized, b, c, h, w) = *src_val;
                    }
                }
            }
        }
    }
    
    double t19_0 = get_time_ms();
    tensor_free(buf_a);
    int32_t concat_19_channels = model->head_convs[2].out_channels + model->head_convs[1].out_channels;  // Layer 18 + Layer 14
    buf_a = tensor_create(1, concat_19_channels, l18_h, l18_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 19\n");
        if (layer14_resized != layer14_output) tensor_free(layer14_resized);
        goto error;
    }
    double t19_alloc = get_time_ms();
    const tensor_t* concat_inputs3[2] = {buf_b, layer14_resized};
    if (concat_forward(concat_inputs3, 2, buf_a) != 0) {
        fprintf(stderr, "Error: Concat forward failed at Layer 19\n");
        fprintf(stderr, "  Input 0 (Layer 18): (%d, %d, %d, %d)\n", 
                buf_b->n, buf_b->c, buf_b->h, buf_b->w);
        fprintf(stderr, "  Input 1 (Layer 14): (%d, %d, %d, %d)\n", 
                layer14_resized->n, layer14_resized->c, layer14_resized->h, layer14_resized->w);
        fprintf(stderr, "  Output: (%d, %d, %d, %d)\n", 
                buf_a->n, buf_a->c, buf_a->h, buf_a->w);
        if (layer14_resized != layer14_output) tensor_free(layer14_resized);
        goto error;
    }
    
    // Free resized tensor if it was created
    if (layer14_resized != layer14_output) {
        tensor_free(layer14_resized);
    }
    double t19_compute = get_time_ms();
    save_feature(model, 19, buf_a);
    double t19_save = get_time_ms();
    print_layer_timing(19, t19_alloc - t19_0, t19_compute - t19_alloc, t19_save - t19_compute);
    // Free layer14_output after use
    tensor_free(layer14_output);
    
    // Layer 20: C3(128->128, n=1, shortcut=False) -> SAVE[20] (P4) for YOLOv5n
    printf("    Layer 20: C3(%d->%d, n=1, shortcut=False)...\n", model->head_c3s[2].c1, model->head_c3s[2].c2);
    fflush(stdout);
    double t20_0 = get_time_ms();
    tensor_free(buf_b);
    buf_b = tensor_create(1, model->head_c3s[2].c2, l18_h, l18_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 20\n");
        goto error;
    }
    double t20_alloc = get_time_ms();
    if (c3_forward(&model->head_c3s[2].block, buf_a, buf_b, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 20\n");
        goto error;
    }
    double t20_compute = get_time_ms();
    save_feature(model, 20, buf_b);
    double t20_save = get_time_ms();
    print_layer_timing(20, t20_alloc - t20_0, t20_compute - t20_alloc, t20_save - t20_compute);
    
    // Output[1] = P4 - resize if needed
    if (output[1]) {
        if (output[1]->h != l18_h || output[1]->w != l18_w) {
            tensor_free(output[1]);
            output[1] = tensor_create(1, model->head_c3s[2].c2, l18_h, l18_w);
            if (!output[1]) goto error;
        }
        tensor_copy(output[1], buf_b);
    }
    
    // Layer 21: Conv(128->128, 3x3, s=2) for YOLOv5n
    printf("    Layer 21: Conv(%d->%d, 3x3, s=2)...\n", model->head_convs[3].in_channels, model->head_convs[3].out_channels);
    fflush(stdout);
    int32_t l21_h = CONV_OUT_D1(l18_h, 3, 2, 1);
    int32_t l21_w = CONV_OUT_D1(l18_w, 3, 2, 1);
    double t21_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_convs[3].out_channels, l21_h, l21_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 21\n");
        goto error;
    }
    double t21_alloc = get_time_ms();
    if (conv2d_fused_bn_silu_forward(&model->head_convs[3].conv,
            model->head_convs[3].is_fused ? NULL : &model->head_convs[3].bn, buf_b, buf_a) != 0) {
        fprintf(stderr, "Error: Conv2D fused forward failed at Layer 21\n");
        goto error;
    }
    double t21_compute = get_time_ms();
    save_feature(model, 21, buf_a);
    double t21_save = get_time_ms();
    print_layer_timing(21, t21_alloc - t21_0, t21_compute - t21_alloc, t21_save - t21_compute);
    
    // Layer 22: Concat([21, 10])
    printf("    Layer 22: Concat([21, 10])...\n");
    fflush(stdout);
    // Verify layer10_output size matches
    if (layer10_output->h != l21_h || layer10_output->w != l21_w) {
        fprintf(stderr, "Error: Layer 10 output size mismatch. Expected (%d, %d), got (%d, %d)\n",
                l21_h, l21_w, layer10_output->h, layer10_output->w);
        goto error;
    }
    
    double t22_0 = get_time_ms();
    tensor_free(buf_b);
    int32_t concat_22_channels = model->head_convs[3].out_channels + model->head_convs[0].out_channels;  // Layer 21 + Layer 10
    buf_b = tensor_create(1, concat_22_channels, l21_h, l21_w);
    if (!buf_b) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 22\n");
        goto error;
    }
    double t22_alloc = get_time_ms();
    const tensor_t* concat_inputs4[2] = {buf_a, layer10_output};
    if (concat_forward(concat_inputs4, 2, buf_b) != 0) {
        fprintf(stderr, "Error: Concat forward failed at Layer 22\n");
        goto error;
    }
    double t22_compute = get_time_ms();
    save_feature(model, 22, buf_b);
    double t22_save = get_time_ms();
    print_layer_timing(22, t22_alloc - t22_0, t22_compute - t22_alloc, t22_save - t22_compute);
    
    // Layer 23: C3(256->256, n=1, shortcut=False) -> SAVE[23] (P5) for YOLOv5n
    printf("    Layer 23: C3(%d->%d, n=1, shortcut=False)...\n", model->head_c3s[3].c1, model->head_c3s[3].c2);
    fflush(stdout);
    double t23_0 = get_time_ms();
    tensor_free(buf_a);
    buf_a = tensor_create(1, model->head_c3s[3].c2, l21_h, l21_w);
    if (!buf_a) {
        fprintf(stderr, "Error: Failed to allocate buffer for Layer 23\n");
        goto error;
    }
    double t23_alloc = get_time_ms();
    if (c3_forward(&model->head_c3s[3].block, buf_b, buf_a, NULL, NULL) != 0) {
        fprintf(stderr, "Error: C3 forward failed at Layer 23\n");
        goto error;
    }
    double t23_compute = get_time_ms();
    save_feature(model, 23, buf_a);
    double t23_save = get_time_ms();
    print_layer_timing(23, t23_alloc - t23_0, t23_compute - t23_alloc, t23_save - t23_compute);
    
    // Output[2] = P5 - resize if needed
    if (output[2]) {
        if (output[2]->h != l21_h || output[2]->w != l21_w) {
            tensor_free(output[2]);
            output[2] = tensor_create(1, model->head_c3s[3].c2, l21_h, l21_w);
            if (!output[2]) goto error;
        }
        tensor_copy(output[2], buf_a);
    }
    
    double forward_end = get_time_ms();
    printf("  Neck completed\n");
    printf("  Total forward pass time: %.2f ms\n", forward_end - forward_start);
    fflush(stdout);
    
    // Save output tensors (P3, P4, P5) if output directory is set
    // These are intermediate feature maps for Detect head, so save to testdata/c/
    if (g_output_dir[0] != '\0' && output[0] && output[1] && output[2]) {
        char filepath[512];
        
        snprintf(filepath, sizeof(filepath), "%s/output_p3.bin", g_output_dir);
        tensor_dump(output[0], filepath);
        
        snprintf(filepath, sizeof(filepath), "%s/output_p4.bin", g_output_dir);
        tensor_dump(output[1], filepath);
        
        snprintf(filepath, sizeof(filepath), "%s/output_p5.bin", g_output_dir);
        tensor_dump(output[2], filepath);
    }
    
    // Cleanup
    tensor_free(buf_a);
    tensor_free(buf_b);
    tensor_free(layer10_output);
    tensor_free(layer13_output);
    
    return 0;
    
error:
    fprintf(stderr, "Error: Forward pass failed during execution\n");
    if (buf_a) tensor_free(buf_a);
    if (buf_b) tensor_free(buf_b);
    if (layer10_output) tensor_free(layer10_output);
    if (layer13_output) tensor_free(layer13_output);
    return -1;
}
