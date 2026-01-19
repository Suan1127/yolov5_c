#ifndef CONCAT_H
#define CONCAT_H

#include "../core/tensor.h"

/**
 * Concatenate tensors along channel dimension (dim=1)
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param output Output tensor (channels = sum of input channels)
 */
int concat_forward(const tensor_t* inputs[], int num_inputs, tensor_t* output);

#endif // CONCAT_H
