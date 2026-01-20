#include "conv2d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int conv2d_init(conv2d_layer_t* layer, int32_t in_channels, const conv2d_params_t* params) {
    if (!layer || !params) return -1;
    
    layer->params = *params;
    layer->in_channels = in_channels;
    
    // Calculate weight size
    size_t weight_size = (size_t)(params->out_channels * in_channels * 
                                   params->kernel_size * params->kernel_size);
    
    layer->weight = (float*)calloc(weight_size, sizeof(float));
    if (!layer->weight) return -1;
    
    // Allocate bias if needed
    if (params->groups == 1) {  // Standard conv has bias
        layer->bias = (float*)calloc(params->out_channels, sizeof(float));
        if (!layer->bias) {
            free(layer->weight);
            return -1;
        }
    } else {
        layer->bias = NULL;
    }
    
    return 0;
}

void conv2d_free(conv2d_layer_t* layer) {
    if (layer) {
        if (layer->weight) free(layer->weight);
        if (layer->bias) free(layer->bias);
        memset(layer, 0, sizeof(conv2d_layer_t));
    }
}

int conv2d_load_weights(conv2d_layer_t* layer, const float* weight_buf, const float* bias_buf) {
    if (!layer || !weight_buf) return -1;
    
    size_t weight_size = (size_t)(layer->params.out_channels * layer->in_channels *
                                   layer->params.kernel_size * layer->params.kernel_size);
    memcpy(layer->weight, weight_buf, weight_size * sizeof(float));
    
    if (bias_buf && layer->bias) {
        memcpy(layer->bias, bias_buf, layer->params.out_channels * sizeof(float));
    }
    
    return 0;
}

// Helper: Get output dimensions
static void conv2d_output_size(int32_t in_h, int32_t in_w, const conv2d_params_t* params,
                               int32_t* out_h, int32_t* out_w) {
    *out_h = (in_h + 2 * params->padding - params->dilation * (params->kernel_size - 1) - 1) / params->stride + 1;
    *out_w = (in_w + 2 * params->padding - params->dilation * (params->kernel_size - 1) - 1) / params->stride + 1;
}

int conv2d_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output) {
    if (!layer || !input || !output) return -1;
    
    // Calculate output dimensions
    int32_t out_h, out_w;
    conv2d_output_size(input->h, input->w, &layer->params, &out_h, &out_w);
    
    // Ensure output tensor has correct size
    // Note: Caller should ensure output tensor is properly allocated
    if (output->n != input->n || output->c != layer->params.out_channels ||
        output->h != out_h || output->w != out_w) {
        fprintf(stderr, "Error: conv2d_forward: Output tensor size mismatch\n");
        fprintf(stderr, "  Expected: (%d, %d, %d, %d), Got: (%d, %d, %d, %d)\n",
                input->n, layer->params.out_channels, out_h, out_w,
                output->n, output->c, output->h, output->w);
        return -1;  // Output tensor size mismatch
    }
    
    // Check input channels match
    if (input->c != layer->in_channels) {
        fprintf(stderr, "Error: conv2d_forward: Input channel mismatch\n");
        fprintf(stderr, "  Expected: %d, Got: %d\n", layer->in_channels, input->c);
        return -1;  // Input channel mismatch
    }
    
    // Check if input and output point to the same memory
    // This is a critical error - input and output should not share memory
    // If they do, we cannot perform convolution correctly
    if (input->data == output->data) {
        fprintf(stderr, "ERROR: conv2d_forward: input and output share the same memory!\n");
        fprintf(stderr, "  input->data=%p, output->data=%p\n", (void*)input->data, (void*)output->data);
        fprintf(stderr, "  input: (%d, %d, %d, %d), output: (%d, %d, %d, %d)\n",
                input->n, input->c, input->h, input->w,
                output->n, output->c, output->h, output->w);
        fprintf(stderr, "  This is a critical error. Caller must provide separate buffers.\n");
        return -1;  // Fail instead of proceeding with corrupted data
    }
    
    tensor_zero(output);
    
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    
    // 1x1 convolution (optimized path)
    if (k == 1 && s == 1 && p == 0) {
        // output[b, oc, h, w] = sum(ic) input[b, ic, h, w] * weight[oc, ic, 0, 0]
        // Weight layout: [out_channels, in_channels, k, k] = [out_channels, in_channels, 1, 1]
        // For 1x1: weight[oc, ic, 0, 0] = weight[oc * (in_channels * 1 * 1) + ic * (1 * 1) + 0 * 1 + 0]
        //         = weight[oc * in_channels + ic]
        static int debug_count = 0;
        for (int32_t b = 0; b < input->n; b++) {
            for (int32_t oc = 0; oc < layer->params.out_channels; oc++) {
                for (int32_t h = 0; h < input->h; h++) {
                    for (int32_t w = 0; w < input->w; w++) {
                        float sum = layer->bias ? layer->bias[oc] : 0.0f;
                        float conv_sum = 0.0f;  // Track conv contribution separately
                        for (int32_t ic = 0; ic < layer->in_channels; ic++) {
                            const float* in_val = tensor_at_const(input, b, ic, h, w);
                            // Weight index: [oc, ic, 0, 0] in [out_channels, in_channels, 1, 1] layout
                            // For 1x1 conv: weight[oc, ic, 0, 0] = weight[oc * in_channels + ic]
                            size_t weight_idx = (size_t)(oc * layer->in_channels + ic);
                            const float* w_val = &layer->weight[weight_idx];
                            float product = (*in_val) * (*w_val);
                            conv_sum += product;
                            sum += product;
                        }
                        // Debug output removed - issue resolved
                        *tensor_at(output, b, oc, h, w) = sum;
                    }
                }
            }
        }
    } else {
        // General convolution (3x3, 6x6, etc.)
        // output[b, oc, oh, ow] = sum(ic, kh, kw) input[b, ic, ih, iw] * weight[oc, ic, kh, kw]
        // where ih = oh * s + kh - p, iw = ow * s + kw - p
        
        for (int32_t b = 0; b < input->n; b++) {
            for (int32_t oc = 0; oc < layer->params.out_channels; oc++) {
                for (int32_t oh = 0; oh < output->h; oh++) {
                    for (int32_t ow = 0; ow < output->w; ow++) {
                        float sum = layer->bias ? layer->bias[oc] : 0.0f;
                        
                        for (int32_t ic = 0; ic < layer->in_channels; ic++) {
                            for (int32_t kh = 0; kh < k; kh++) {
                                for (int32_t kw = 0; kw < k; kw++) {
                                    int32_t ih = oh * s + kh - p;
                                    int32_t iw = ow * s + kw - p;
                                    
                                    // Check bounds
                                    if (ih >= 0 && ih < input->h && iw >= 0 && iw < input->w) {
                                        const float* in_val = tensor_at_const(input, b, ic, ih, iw);
                                        const float* w_val = &layer->weight[
                                            oc * (layer->in_channels * k * k) +
                                            ic * (k * k) +
                                            kh * k + kw
                                        ];
                                        sum += (*in_val) * (*w_val);
                                    }
                                }
                            }
                        }
                        
                        *tensor_at(output, b, oc, oh, ow) = sum;
                    }
                }
            }
        }
    }
    
    return 0;
}
