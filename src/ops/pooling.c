#include "pooling.h"
#include <string.h>
#include <float.h>

int maxpool2d_forward(const maxpool2d_params_t* params, const tensor_t* input, tensor_t* output) {
    if (!params || !input || !output) return -1;
    
    int32_t k = params->kernel_size;
    int32_t s = params->stride;
    int32_t p = params->padding;
    
    // Calculate output size
    int32_t out_h = (input->h + 2 * p - k) / s + 1;
    int32_t out_w = (input->w + 2 * p - k) / s + 1;
    
    if (output->n != input->n || output->c != input->c ||
        output->h != out_h || output->w != out_w) {
        return -1;
    }
    
    // MaxPool2D: output[b, c, oh, ow] = max(input[b, c, ih:ih+k, iw:iw+k])
    for (int32_t b = 0; b < input->n; b++) {
        for (int32_t c = 0; c < input->c; c++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    
                    for (int32_t kh = 0; kh < k; kh++) {
                        for (int32_t kw = 0; kw < k; kw++) {
                            int32_t ih = oh * s + kh - p;
                            int32_t iw = ow * s + kw - p;
                            
                            if (ih >= 0 && ih < input->h && iw >= 0 && iw < input->w) {
                                const float* in_val = tensor_at_const(input, b, c, ih, iw);
                                if (*in_val > max_val) {
                                    max_val = *in_val;
                                }
                            }
                        }
                    }
                    
                    *tensor_at(output, b, c, oh, ow) = max_val;
                }
            }
        }
    }
    
    return 0;
}
