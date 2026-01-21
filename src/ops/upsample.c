#include "upsample.h"
#include <stdio.h>
#include <string.h>

int upsample_forward(const upsample_params_t* params, const tensor_t* input, tensor_t* output) {
    if (!params || !input || !output) return -1;
    
    if (params->scale_factor != 2 || strcmp(params->mode, "nearest") != 0) {
        return -1;  // Only support nearest ×2 for now
    }
    
    // Output size should be 2x input size
    if (output->n != input->n || output->c != input->c ||
        output->h != input->h * 2 || output->w != input->w * 2) {
        fprintf(stderr, "Error: upsample_forward: Size mismatch\n");
        fprintf(stderr, "  Input: (%d, %d, %d, %d)\n", input->n, input->c, input->h, input->w);
        fprintf(stderr, "  Output: (%d, %d, %d, %d)\n", output->n, output->c, output->h, output->w);
        fprintf(stderr, "  Expected output: (%d, %d, %d, %d)\n", input->n, input->c, input->h * 2, input->w * 2);
        return -1;
    }
    
    // Nearest neighbor upsampling: replicate each pixel 2×2
    for (int32_t b = 0; b < input->n; b++) {
        for (int32_t c = 0; c < input->c; c++) {
            for (int32_t ih = 0; ih < input->h; ih++) {
                for (int32_t iw = 0; iw < input->w; iw++) {
                    const float* in_val = tensor_at_const(input, b, c, ih, iw);
                    float val = *in_val;
                    
                    // Replicate to 2×2 block
                    *tensor_at(output, b, c, ih * 2, iw * 2) = val;
                    *tensor_at(output, b, c, ih * 2, iw * 2 + 1) = val;
                    *tensor_at(output, b, c, ih * 2 + 1, iw * 2) = val;
                    *tensor_at(output, b, c, ih * 2 + 1, iw * 2 + 1) = val;
                }
            }
        }
    }
    
    return 0;
}
