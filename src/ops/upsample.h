#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include "../core/tensor.h"

/**
 * Upsample parameters
 */
typedef struct {
    int32_t scale_factor;  // e.g., 2 for 2x upsampling
    const char* mode;      // "nearest" or "bilinear"
} upsample_params_t;

/**
 * Upsample forward pass (nearest neighbor)
 */
int upsample_forward(const upsample_params_t* params, const tensor_t* input, tensor_t* output);

#endif // UPSAMPLE_H
