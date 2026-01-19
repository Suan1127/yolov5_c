#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

#include "../core/tensor.h"

/**
 * BatchNorm2D parameters
 */
typedef struct {
    int32_t num_features;  // Number of channels
    float eps;              // Epsilon (default: 1e-5)
    float momentum;         // Momentum (default: 0.1, not used in inference)
} batchnorm2d_params_t;

/**
 * BatchNorm2D layer
 */
typedef struct {
    batchnorm2d_params_t params;
    float* weight;         // [num_features] (gamma)
    float* bias;            // [num_features] (beta)
    float* running_mean;    // [num_features]
    float* running_var;     // [num_features]
} batchnorm2d_layer_t;

/**
 * Initialize BatchNorm2D layer
 */
int batchnorm2d_init(batchnorm2d_layer_t* layer, int32_t num_features, const batchnorm2d_params_t* params);

/**
 * Free BatchNorm2D layer
 */
void batchnorm2d_free(batchnorm2d_layer_t* layer);

/**
 * Forward pass: output = batchnorm2d(input)
 * During inference: output = (input - running_mean) / sqrt(running_var + eps) * weight + bias
 */
int batchnorm2d_forward(const batchnorm2d_layer_t* layer, const tensor_t* input, tensor_t* output);

/**
 * Load weights from buffers
 */
int batchnorm2d_load_weights(batchnorm2d_layer_t* layer, 
                              const float* weight_buf, const float* bias_buf,
                              const float* running_mean_buf, const float* running_var_buf);

#endif // BATCHNORM2D_H
