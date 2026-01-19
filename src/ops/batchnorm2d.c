#include "batchnorm2d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int batchnorm2d_init(batchnorm2d_layer_t* layer, int32_t num_features, const batchnorm2d_params_t* params) {
    if (!layer || !params) return -1;
    
    layer->params = *params;
    layer->params.num_features = num_features;
    
    layer->weight = (float*)calloc(num_features, sizeof(float));
    layer->bias = (float*)calloc(num_features, sizeof(float));
    layer->running_mean = (float*)calloc(num_features, sizeof(float));
    layer->running_var = (float*)calloc(num_features, sizeof(float));
    
    if (!layer->weight || !layer->bias || !layer->running_mean || !layer->running_var) {
        batchnorm2d_free(layer);
        return -1;
    }
    
    // Initialize weight to 1.0, bias to 0.0
    for (int i = 0; i < num_features; i++) {
        layer->weight[i] = 1.0f;
    }
    
    return 0;
}

void batchnorm2d_free(batchnorm2d_layer_t* layer) {
    if (layer) {
        if (layer->weight) free(layer->weight);
        if (layer->bias) free(layer->bias);
        if (layer->running_mean) free(layer->running_mean);
        if (layer->running_var) free(layer->running_var);
        memset(layer, 0, sizeof(batchnorm2d_layer_t));
    }
}

int batchnorm2d_load_weights(batchnorm2d_layer_t* layer, 
                              const float* weight_buf, const float* bias_buf,
                              const float* running_mean_buf, const float* running_var_buf) {
    if (!layer) return -1;
    
    int32_t n = layer->params.num_features;
    
    if (weight_buf) {
        memcpy(layer->weight, weight_buf, n * sizeof(float));
    }
    if (bias_buf) {
        memcpy(layer->bias, bias_buf, n * sizeof(float));
    }
    if (running_mean_buf) {
        memcpy(layer->running_mean, running_mean_buf, n * sizeof(float));
    }
    if (running_var_buf) {
        memcpy(layer->running_var, running_var_buf, n * sizeof(float));
    }
    
    return 0;
}

int batchnorm2d_forward(const batchnorm2d_layer_t* layer, const tensor_t* input, tensor_t* output) {
    if (!layer || !input || !output) return -1;
    
    // Ensure output has same shape as input
    if (output->n != input->n || output->c != input->c || 
        output->h != input->h || output->w != input->w) {
        return -1;
    }
    
    if (input->c != layer->params.num_features) {
        return -1;
    }
    
    float eps = layer->params.eps;
    
    // BatchNorm2D: output = (input - mean) / sqrt(var + eps) * weight + bias
    for (int32_t b = 0; b < input->n; b++) {
        for (int32_t c = 0; c < input->c; c++) {
            float weight = layer->weight[c];
            float bias = layer->bias[c];
            float mean = layer->running_mean[c];
            float var = layer->running_var[c];
            float inv_std = 1.0f / sqrtf(var + eps);
            
            for (int32_t h = 0; h < input->h; h++) {
                for (int32_t w = 0; w < input->w; w++) {
                    const float* in_val = tensor_at_const(input, b, c, h, w);
                    float normalized = (*in_val - mean) * inv_std;
                    *tensor_at(output, b, c, h, w) = normalized * weight + bias;
                }
            }
        }
    }
    
    return 0;
}
