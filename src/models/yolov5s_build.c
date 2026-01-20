#include "yolov5s_build.h"
#include "../core/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
// Note: JSON parsing not needed for basic build - model_meta.json can be loaded later if needed

// Helper: make divisible by 8
static int32_t make_divisible(int32_t x, int32_t divisor) {
    return (int32_t)(ceil((double)x / divisor) * divisor);
}

// Helper: calculate actual channels
static int32_t get_actual_channels(int32_t base_channels, float width_multiple) {
    return make_divisible((int32_t)(base_channels * width_multiple), 8);
}

// Helper: calculate actual repeats
static int32_t get_actual_repeats(int32_t base_repeats, float depth_multiple) {
    int32_t result = (int32_t)round(base_repeats * depth_multiple);
    return result > 0 ? result : 1;
}

// Helper: Load Conv+BN layer weights
int load_conv_bn_layer(conv2d_layer_t* conv, batchnorm2d_layer_t* bn,
                       weights_loader_t* loader, const char* prefix,
                       int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t stride, int32_t padding) {
    if (!conv || !bn || !loader || !prefix) return -1;
    
    char name[256];
    int32_t shape[4];
    int num_dims;
    
    // Initialize conv layer
    conv2d_params_t conv_params = {
        .out_channels = out_channels,
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(conv, in_channels, &conv_params) != 0) return -1;
    
    // Load conv weights
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (w) {
        conv2d_load_weights(conv, w, NULL);
    }
    
    // Initialize BN layer
    batchnorm2d_params_t bn_params = {
        .num_features = out_channels,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(bn, out_channels, &bn_params) != 0) {
        conv2d_free(conv);
        return -1;
    }
    
    // Load BN weights
    snprintf(name, sizeof(name), "%s.bn.weight", prefix);
    float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.bias", prefix);
    float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.running_mean", prefix);
    float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.running_var", prefix);
    float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
    
    if (bn_w && bn_b && bn_mean && bn_var) {
        batchnorm2d_load_weights(bn, bn_w, bn_b, bn_mean, bn_var);
    }
    
    return 0;
}

yolov5s_model_t* yolov5s_build(const char* weights_path, const char* model_meta_path) {
    if (!weights_path) return NULL;
    
    yolov5s_model_t* model = (yolov5s_model_t*)calloc(1, sizeof(yolov5s_model_t));
    if (!model) return NULL;
    
    // Default parameters (YOLOv5s)
    model->depth_multiple = 0.33f;
    model->width_multiple = 0.50f;
    model->num_classes = 80;
    model->input_size = 640;
    
    // TODO: Load model_meta.json if provided to get actual parameters
    // For now, use defaults
    
    // Load weights
    printf("Loading weights from: %s\n", weights_path);
    model->weights = weights_loader_create(weights_path);
    if (!model->weights) {
        fprintf(stderr, "Error: Failed to load weights from %s\n", weights_path);
        fprintf(stderr, "Check if file exists and is readable\n");
        free(model);
        return NULL;
    }
    printf("Weights loaded successfully (size: %zu bytes)\n", model->weights->size);
    
    // Initialize saved features
    for (int i = 0; i < 12; i++) {
        model->saved_features[i] = NULL;
    }
    
    // Load weights
    if (yolov5s_load_weights(model) != 0) {
        yolov5s_free(model);
        return NULL;
    }
    
    return model;
}

void yolov5s_free(yolov5s_model_t* model) {
    if (!model) return;
    
    // Free backbone conv layers
    for (int i = 0; i < 5; i++) {
        conv2d_free(&model->backbone_convs[i].conv);
        batchnorm2d_free(&model->backbone_convs[i].bn);
    }
    
    // Free backbone C3 blocks
    for (int i = 0; i < 4; i++) {
        c3_free(&model->backbone_c3s[i].block);
    }
    
    // Free SPPF
    sppf_free(&model->sppf);
    
    // Free head conv layers
    for (int i = 0; i < 4; i++) {
        conv2d_free(&model->head_convs[i].conv);
        batchnorm2d_free(&model->head_convs[i].bn);
    }
    
    // Free head C3 blocks
    for (int i = 0; i < 4; i++) {
        c3_free(&model->head_c3s[i].block);
    }
    
    // Free saved features
    for (int i = 0; i < 12; i++) {
        if (model->saved_features[i]) {
            tensor_free(model->saved_features[i]);
        }
    }
    
    // Free weights loader
    if (model->weights) {
        weights_loader_free(model->weights);
    }
    
    free(model);
}

int yolov5s_load_weights(yolov5s_model_t* model) {
    if (!model || !model->weights) return -1;
    
    float depth_multiple = model->depth_multiple;
    float width_multiple = model->width_multiple;
    char name[256];
    
    // ========== Backbone ==========
    
    // Layer 0: Conv(3→32, 6×6, s=2, p=2)
    model->backbone_convs[0].in_channels = 3;
    model->backbone_convs[0].out_channels = get_actual_channels(64, width_multiple);  // 32
    model->backbone_convs[0].kernel_size = 6;
    model->backbone_convs[0].stride = 2;
    model->backbone_convs[0].padding = 2;
    snprintf(name, sizeof(name), "model.0");
    if (load_conv_bn_layer(&model->backbone_convs[0].conv, &model->backbone_convs[0].bn,
                          model->weights, name,
                          model->backbone_convs[0].in_channels,
                          model->backbone_convs[0].out_channels,
                          model->backbone_convs[0].kernel_size,
                          model->backbone_convs[0].stride,
                          model->backbone_convs[0].padding) != 0) {
        return -1;
    }
    
    // Layer 1: Conv(32→64, 3×3, s=2)
    model->backbone_convs[1].in_channels = 32;
    model->backbone_convs[1].out_channels = get_actual_channels(128, width_multiple);  // 64
    model->backbone_convs[1].kernel_size = 3;
    model->backbone_convs[1].stride = 2;
    model->backbone_convs[1].padding = 1;
    snprintf(name, sizeof(name), "model.1");
    if (load_conv_bn_layer(&model->backbone_convs[1].conv, &model->backbone_convs[1].bn,
                          model->weights, name,
                          model->backbone_convs[1].in_channels,
                          model->backbone_convs[1].out_channels,
                          model->backbone_convs[1].kernel_size,
                          model->backbone_convs[1].stride,
                          model->backbone_convs[1].padding) != 0) {
        return -1;
    }
    
    // Layer 2: C3(64→64, n=1)
    model->backbone_c3s[0].c1 = 64;
    model->backbone_c3s[0].c2 = 64;
    model->backbone_c3s[0].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[0].shortcut = 1;
    if (c3_init(&model->backbone_c3s[0].block, 64, 64, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.2");
    if (c3_load_weights(&model->backbone_c3s[0].block, model->weights, name) != 0) {
        c3_free(&model->backbone_c3s[0].block);
        return -1;
    }
    
    // Layer 3: Conv(64→128, 3×3, s=2)
    model->backbone_convs[2].in_channels = 64;
    model->backbone_convs[2].out_channels = get_actual_channels(256, width_multiple);  // 128
    model->backbone_convs[2].kernel_size = 3;
    model->backbone_convs[2].stride = 2;
    model->backbone_convs[2].padding = 1;
    snprintf(name, sizeof(name), "model.3");
    if (load_conv_bn_layer(&model->backbone_convs[2].conv, &model->backbone_convs[2].bn,
                          model->weights, name,
                          model->backbone_convs[2].in_channels,
                          model->backbone_convs[2].out_channels,
                          model->backbone_convs[2].kernel_size,
                          model->backbone_convs[2].stride,
                          model->backbone_convs[2].padding) != 0) {
        return -1;
    }
    
    // Layer 4: C3(128→128, n=2)
    model->backbone_c3s[1].c1 = 128;
    model->backbone_c3s[1].c2 = 128;
    model->backbone_c3s[1].n = get_actual_repeats(6, depth_multiple);  // 2
    model->backbone_c3s[1].shortcut = 1;
    if (c3_init(&model->backbone_c3s[1].block, 128, 128, 2, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.4");
    if (c3_load_weights(&model->backbone_c3s[1].block, model->weights, name) != 0) {
        c3_free(&model->backbone_c3s[1].block);
        return -1;
    }
    
    // Layer 5: Conv(128→256, 3×3, s=2)
    model->backbone_convs[3].in_channels = 128;
    model->backbone_convs[3].out_channels = get_actual_channels(512, width_multiple);  // 256
    model->backbone_convs[3].kernel_size = 3;
    model->backbone_convs[3].stride = 2;
    model->backbone_convs[3].padding = 1;
    snprintf(name, sizeof(name), "model.5");
    if (load_conv_bn_layer(&model->backbone_convs[3].conv, &model->backbone_convs[3].bn,
                          model->weights, name,
                          model->backbone_convs[3].in_channels,
                          model->backbone_convs[3].out_channels,
                          model->backbone_convs[3].kernel_size,
                          model->backbone_convs[3].stride,
                          model->backbone_convs[3].padding) != 0) {
        return -1;
    }
    
    // Layer 6: C3(256→256, n=3)
    model->backbone_c3s[2].c1 = 256;
    model->backbone_c3s[2].c2 = 256;
    model->backbone_c3s[2].n = get_actual_repeats(9, depth_multiple);  // 3
    model->backbone_c3s[2].shortcut = 1;
    if (c3_init(&model->backbone_c3s[2].block, 256, 256, 3, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.6");
    if (c3_load_weights(&model->backbone_c3s[2].block, model->weights, name) != 0) {
        c3_free(&model->backbone_c3s[2].block);
        return -1;
    }
    
    // Layer 7: Conv(256→512, 3×3, s=2)
    model->backbone_convs[4].in_channels = 256;
    model->backbone_convs[4].out_channels = get_actual_channels(1024, width_multiple);  // 512
    model->backbone_convs[4].kernel_size = 3;
    model->backbone_convs[4].stride = 2;
    model->backbone_convs[4].padding = 1;
    snprintf(name, sizeof(name), "model.7");
    if (load_conv_bn_layer(&model->backbone_convs[4].conv, &model->backbone_convs[4].bn,
                          model->weights, name,
                          model->backbone_convs[4].in_channels,
                          model->backbone_convs[4].out_channels,
                          model->backbone_convs[4].kernel_size,
                          model->backbone_convs[4].stride,
                          model->backbone_convs[4].padding) != 0) {
        return -1;
    }
    
    // Layer 8: C3(512→512, n=1)
    model->backbone_c3s[3].c1 = 512;
    model->backbone_c3s[3].c2 = 512;
    model->backbone_c3s[3].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[3].shortcut = 1;
    if (c3_init(&model->backbone_c3s[3].block, 512, 512, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.8");
    if (c3_load_weights(&model->backbone_c3s[3].block, model->weights, name) != 0) {
        c3_free(&model->backbone_c3s[3].block);
        return -1;
    }
    
    // Layer 9: SPPF(512→512, k=5)
    if (sppf_init(&model->sppf, 512, 512, 5) != 0) return -1;
    snprintf(name, sizeof(name), "model.9");
    if (sppf_load_weights(&model->sppf, model->weights, name) != 0) {
        sppf_free(&model->sppf);
        return -1;
    }
    
    // ========== Head ==========
    
    // Layer 10: Conv(512→256, 1×1)
    model->head_convs[0].in_channels = 512;
    model->head_convs[0].out_channels = 256;
    snprintf(name, sizeof(name), "model.10");
    if (load_conv_bn_layer(&model->head_convs[0].conv, &model->head_convs[0].bn,
                          model->weights, name,
                          model->head_convs[0].in_channels,
                          model->head_convs[0].out_channels,
                          1, 1, 0) != 0) {
        return -1;
    }
    
    // Layer 13: C3(512→256, n=1, shortcut=False)
    model->head_c3s[0].c1 = 512;  // From concat of layer 11 (256) and layer 6 (256)
    model->head_c3s[0].c2 = 256;
    model->head_c3s[0].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[0].shortcut = 0;
    if (c3_init(&model->head_c3s[0].block, 512, 256, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.13");
    if (c3_load_weights(&model->head_c3s[0].block, model->weights, name) != 0) {
        c3_free(&model->head_c3s[0].block);
        return -1;
    }
    
    // Layer 14: Conv(256→128, 1×1)
    model->head_convs[1].in_channels = 256;
    model->head_convs[1].out_channels = 128;
    snprintf(name, sizeof(name), "model.14");
    if (load_conv_bn_layer(&model->head_convs[1].conv, &model->head_convs[1].bn,
                          model->weights, name,
                          model->head_convs[1].in_channels,
                          model->head_convs[1].out_channels,
                          1, 1, 0) != 0) {
        return -1;
    }
    
    // Layer 17: C3(256→128, n=1, shortcut=False)
    model->head_c3s[1].c1 = 256;  // From concat of layer 15 (128) and layer 4 (128)
    model->head_c3s[1].c2 = 128;
    model->head_c3s[1].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[1].shortcut = 0;
    if (c3_init(&model->head_c3s[1].block, 256, 128, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.17");
    if (c3_load_weights(&model->head_c3s[1].block, model->weights, name) != 0) {
        c3_free(&model->head_c3s[1].block);
        return -1;
    }
    
    // Layer 18: Conv(128→128, 3×3, s=2)
    model->head_convs[2].in_channels = 128;
    model->head_convs[2].out_channels = 128;
    snprintf(name, sizeof(name), "model.18");
    if (load_conv_bn_layer(&model->head_convs[2].conv, &model->head_convs[2].bn,
                          model->weights, name,
                          model->head_convs[2].in_channels,
                          model->head_convs[2].out_channels,
                          3, 2, 1) != 0) {
        return -1;
    }
    
    // Layer 20: C3(256→256, n=1, shortcut=False)
    model->head_c3s[2].c1 = 256;  // From concat of layer 18 (128) and layer 13 (256)
    model->head_c3s[2].c2 = 256;
    model->head_c3s[2].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[2].shortcut = 0;
    if (c3_init(&model->head_c3s[2].block, 256, 256, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.20");
    if (c3_load_weights(&model->head_c3s[2].block, model->weights, name) != 0) {
        c3_free(&model->head_c3s[2].block);
        return -1;
    }
    
    // Layer 21: Conv(256→256, 3×3, s=2)
    model->head_convs[3].in_channels = 256;
    model->head_convs[3].out_channels = 256;
    snprintf(name, sizeof(name), "model.21");
    if (load_conv_bn_layer(&model->head_convs[3].conv, &model->head_convs[3].bn,
                          model->weights, name,
                          model->head_convs[3].in_channels,
                          model->head_convs[3].out_channels,
                          3, 2, 1) != 0) {
        return -1;
    }
    
    // Layer 23: C3(512→512, n=1, shortcut=False)
    model->head_c3s[3].c1 = 512;  // From concat of layer 21 (256) and layer 10 (256)
    model->head_c3s[3].c2 = 512;
    model->head_c3s[3].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[3].shortcut = 0;
    if (c3_init(&model->head_c3s[3].block, 512, 512, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.23");
    if (c3_load_weights(&model->head_c3s[3].block, model->weights, name) != 0) {
        c3_free(&model->head_c3s[3].block);
        return -1;
    }
    
    printf("YOLOv5s model loaded successfully\n");
    return 0;
}
