#ifndef YOLOV5S_INFER_UTILS_H
#define YOLOV5S_INFER_UTILS_H

#include "../core/tensor.h"
#include "../ops/conv2d.h"

/**
 * Calculate output size after convolution
 */
static inline int32_t conv_output_size(int32_t in_size, int32_t kernel_size, 
                                        int32_t stride, int32_t padding, int32_t dilation) {
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

/**
 * Calculate output size after pooling
 */
static inline int32_t pool_output_size(int32_t in_size, int32_t kernel_size,
                                       int32_t stride, int32_t padding) {
    return (in_size + 2 * padding - kernel_size) / stride + 1;
}

#endif // YOLOV5S_INFER_UTILS_H
