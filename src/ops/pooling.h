#ifndef POOLING_H
#define POOLING_H

#include "../core/tensor.h"

/**
 * MaxPool2D parameters
 */
typedef struct {
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
} maxpool2d_params_t;

/**
 * MaxPool2D forward pass
 */
int maxpool2d_forward(const maxpool2d_params_t* params, const tensor_t* input, tensor_t* output);

#endif // POOLING_H
