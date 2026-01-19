#ifndef CONV2D_H
#define CONV2D_H

#include "../core/tensor.h"

/**
 * 2D Convolution parameters
 */
typedef struct {
    int32_t out_channels;
    int32_t kernel_size;   // Assuming square kernel (k×k)
    int32_t stride;
    int32_t padding;
    int32_t groups;        // For depthwise conv (default: 1)
    int32_t dilation;      // Default: 1
} conv2d_params_t;

/**
 * Convolution layer with weights and bias
 */
typedef struct {
    conv2d_params_t params;
    float* weight;         // [out_channels, in_channels, k, k]
    float* bias;           // [out_channels] or NULL
    int32_t in_channels;
} conv2d_layer_t;

/**
 * Initialize convolution layer
 */
int conv2d_init(conv2d_layer_t* layer, int32_t in_channels, const conv2d_params_t* params);

/**
 * Free convolution layer
 */
void conv2d_free(conv2d_layer_t* layer);

/**
 * Forward pass: output = conv2d(input)
 */
int conv2d_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output);

/**
 * Load weights from buffer
 * @param layer Layer to load weights into
 * @param weight_buf Weight buffer [out_c, in_c, k, k]
 * @param bias_buf Bias buffer [out_c] or NULL
 */
int conv2d_load_weights(conv2d_layer_t* layer, const float* weight_buf, const float* bias_buf);

#endif // CONV2D_H
