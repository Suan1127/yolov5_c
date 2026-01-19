#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdint.h>
#include <stddef.h>

/**
 * Weights loader for loading weights from weights.bin file
 * Uses weights_map.json to locate each layer's weights
 */

typedef struct {
    float* data;           // Pointer to weights buffer
    size_t size;           // Total size in bytes
    const char* map_path;  // Path to weights_map.json (for reference)
} weights_loader_t;

/**
 * Load weights from binary file
 * @param weights_path Path to weights.bin
 * @return Loaded weights loader, or NULL on failure
 */
weights_loader_t* weights_loader_create(const char* weights_path);

/**
 * Free weights loader
 */
void weights_loader_free(weights_loader_t* loader);

/**
 * Get pointer to weight tensor by name (from weights_map.json)
 * @param loader Weights loader
 * @param name Layer name (e.g., "model.0.conv.weight")
 * @param shape Output shape [out_channels, in_channels, h, w] or [channels]
 * @param num_dims Output number of dimensions
 * @return Pointer to weight data, or NULL if not found
 */
float* weights_loader_get(weights_loader_t* loader, const char* name, 
                          int32_t* shape, int* num_dims);

/**
 * Get offset for a weight tensor by name
 * @param loader Weights loader
 * @param name Layer name
 * @return Offset in bytes, or -1 if not found
 */
int64_t weights_loader_get_offset(weights_loader_t* loader, const char* name);

#endif // WEIGHTS_LOADER_H
