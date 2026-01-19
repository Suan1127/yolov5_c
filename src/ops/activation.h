#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../core/tensor.h"

/**
 * SiLU activation: x * sigmoid(x)
 */
void activation_silu(tensor_t* t);

/**
 * Sigmoid function
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#endif // ACTIVATION_H
