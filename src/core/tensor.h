#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>

/**
 * Tensor structure (NCHW layout)
 * Data is stored in row-major order: [N][C][H][W]
 */
typedef struct {
    float* data;        // Pointer to data buffer
    int32_t n;          // Batch size
    int32_t c;          // Channels
    int32_t h;          // Height
    int32_t w;          // Width
    size_t capacity;    // Allocated capacity (in elements)
} tensor_t;

/**
 * Create a new tensor with specified dimensions
 * @param n Batch size
 * @param c Channels
 * @param h Height
 * @param w Width
 * @return Pointer to allocated tensor, or NULL on failure
 */
tensor_t* tensor_create(int32_t n, int32_t c, int32_t h, int32_t w);

/**
 * Free tensor memory
 */
void tensor_free(tensor_t* t);

/**
 * Get total number of elements
 */
static inline size_t tensor_size(const tensor_t* t) {
    return (size_t)(t->n * t->c * t->h * t->w);
}

/**
 * Get element at position (n, c, h, w)
 */
static inline float* tensor_at(tensor_t* t, int32_t n, int32_t c, int32_t h, int32_t w) {
    return &t->data[n * (t->c * t->h * t->w) + c * (t->h * t->w) + h * t->w + w];
}

/**
 * Get const element at position (n, c, h, w)
 */
static inline const float* tensor_at_const(const tensor_t* t, int32_t n, int32_t c, int32_t h, int32_t w) {
    return &t->data[n * (t->c * t->h * t->w) + c * (t->h * t->w) + h * t->w + w];
}

/**
 * Fill tensor with zeros
 */
void tensor_zero(tensor_t* t);

/**
 * Fill tensor with a constant value
 */
void tensor_fill(tensor_t* t, float value);

/**
 * Copy tensor data
 */
void tensor_copy(tensor_t* dst, const tensor_t* src);

/**
 * Dump tensor to file (binary format)
 */
int tensor_dump(const tensor_t* t, const char* filename);

/**
 * Load tensor from file (binary format)
 */
tensor_t* tensor_load(const char* filename);

#endif // TENSOR_H
