#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

tensor_t* tensor_create(int32_t n, int32_t c, int32_t h, int32_t w) {
    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    if (!t) return NULL;
    
    size_t size = (size_t)(n * c * h * w);
    t->data = (float*)calloc(size, sizeof(float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    
    t->n = n;
    t->c = c;
    t->h = h;
    t->w = w;
    t->capacity = size;
    
    return t;
}

void tensor_free(tensor_t* t) {
    if (t) {
        if (t->data) {
            free(t->data);
        }
        free(t);
    }
}

void tensor_zero(tensor_t* t) {
    if (t && t->data) {
        size_t size = tensor_size(t);
        memset(t->data, 0, size * sizeof(float));
    }
}

void tensor_fill(tensor_t* t, float value) {
    if (t && t->data) {
        size_t size = tensor_size(t);
        for (size_t i = 0; i < size; i++) {
            t->data[i] = value;
        }
    }
}

void tensor_copy(tensor_t* dst, const tensor_t* src) {
    if (!dst || !src || !dst->data || !src->data) return;
    
    size_t size = tensor_size(src);
    if (size != tensor_size(dst)) {
        // Resize if needed
        if (dst->capacity < size) {
            free(dst->data);
            dst->data = (float*)malloc(size * sizeof(float));
            dst->capacity = size;
        }
        dst->n = src->n;
        dst->c = src->c;
        dst->h = src->h;
        dst->w = src->w;
    }
    
    memcpy(dst->data, src->data, size * sizeof(float));
}

int tensor_dump(const tensor_t* t, const char* filename) {
    if (!t || !filename) return -1;
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;
    
    // Write header: n, c, h, w (as int32_t)
    int32_t dims[4] = {t->n, t->c, t->h, t->w};
    if (fwrite(dims, sizeof(int32_t), 4, fp) != 4) {
        fclose(fp);
        return -1;
    }
    
    // Write data
    size_t size = tensor_size(t);
    if (fwrite(t->data, sizeof(float), size, fp) != size) {
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}

tensor_t* tensor_load(const char* filename) {
    if (!filename) return NULL;
    
    FILE* fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    // Read header
    int32_t dims[4];
    if (fread(dims, sizeof(int32_t), 4, fp) != 4) {
        fclose(fp);
        return NULL;
    }
    
    tensor_t* t = tensor_create(dims[0], dims[1], dims[2], dims[3]);
    if (!t) {
        fclose(fp);
        return NULL;
    }
    
    // Read data
    size_t size = tensor_size(t);
    if (fread(t->data, sizeof(float), size, fp) != size) {
        tensor_free(t);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return t;
}
