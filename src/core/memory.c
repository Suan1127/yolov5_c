#include "memory.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <stdalign.h>
#define ALIGNED_ALLOC(size, alignment) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

arena_t* arena_create(size_t capacity) {
    arena_t* arena = (arena_t*)malloc(sizeof(arena_t));
    if (!arena) return NULL;
    
    // Align to 16 bytes for SIMD
    size_t align_size = (capacity + 15) & ~15;
    arena->base = (char*)ALIGNED_ALLOC(align_size, 16);
    if (!arena->base) {
        free(arena);
        return NULL;
    }
    
    arena->size = align_size;
    arena->used = 0;
    
    return arena;
}

void arena_destroy(arena_t* arena) {
    if (arena) {
        if (arena->base) {
            ALIGNED_FREE(arena->base);
        }
        free(arena);
    }
}

void* arena_alloc(arena_t* arena, size_t size) {
    if (!arena || !arena->base) return NULL;
    
    // Align to 16 bytes
    size_t align_size = (size + 15) & ~15;
    
    if (arena->used + align_size > arena->size) {
        return NULL;  // Out of memory
    }
    
    void* ptr = arena->base + arena->used;
    arena->used += align_size;
    
    return ptr;
}

void arena_reset(arena_t* arena) {
    if (arena) {
        arena->used = 0;
    }
}
