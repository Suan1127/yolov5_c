#include "concat.h"
#include <string.h>
#include <assert.h>

int concat_forward(const tensor_t* inputs[], int num_inputs, tensor_t* output) {
    if (!inputs || num_inputs < 1 || !output) return -1;
    
    // Verify all inputs have same N, H, W
    int32_t n = inputs[0]->n;
    int32_t h = inputs[0]->h;
    int32_t w = inputs[0]->w;
    int32_t total_channels = 0;
    
    for (int i = 0; i < num_inputs; i++) {
        if (!inputs[i]) return -1;
        if (inputs[i]->n != n || inputs[i]->h != h || inputs[i]->w != w) {
            return -1;  // Shape mismatch
        }
        total_channels += inputs[i]->c;
    }
    
    // Verify output shape
    if (output->n != n || output->c != total_channels || 
        output->h != h || output->w != w) {
        return -1;
    }
    
    // Concatenate along channel dimension
    int32_t out_c = 0;
    for (int i = 0; i < num_inputs; i++) {
        int32_t in_c = inputs[i]->c;
        
        for (int32_t b = 0; b < n; b++) {
            for (int32_t c = 0; c < in_c; c++) {
                for (int32_t h_idx = 0; h_idx < h; h_idx++) {
                    for (int32_t w_idx = 0; w_idx < w; w_idx++) {
                        const float* in_val = tensor_at_const(inputs[i], b, c, h_idx, w_idx);
                        *tensor_at(output, b, out_c + c, h_idx, w_idx) = *in_val;
                    }
                }
            }
        }
        out_c += in_c;
    }
    
    return 0;
}
