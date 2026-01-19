#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

/**
 * Arena allocator for efficient memory management
 * All allocations are freed together when arena is destroyed
 */
typedef struct {
    char* base;
    size_t size;
    size_t used;
} arena_t;

/**
 * Create a new arena with specified capacity
 */
arena_t* arena_create(size_t capacity);

/**
 * Destroy arena and free all memory
 */
void arena_destroy(arena_t* arena);

/**
 * Allocate memory from arena
 */
void* arena_alloc(arena_t* arena, size_t size);

/**
 * Reset arena (free all allocations, keep memory)
 */
void arena_reset(arena_t* arena);

/**
 * Get current usage
 */
static inline size_t arena_used(const arena_t* arena) {
    return arena ? arena->used : 0;
}

#endif // MEMORY_H
