#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define strdup _strdup
#endif

// Simple JSON parser for weights_map.json
// Note: This is a minimal implementation. For production, consider using a proper JSON library.

typedef struct {
    char* name;
    int64_t offset;
    int32_t shape[4];
    int num_dims;
} weight_entry_t;

typedef struct {
    weight_entry_t* entries;
    int num_entries;
    int capacity;
} weight_map_t;

static weight_map_t* parse_weights_map(const char* map_path) {
    FILE* fp = fopen(map_path, "r");
    if (!fp) return NULL;
    
    // Simple parsing - find "offset" and "shape" values
    // This is a basic implementation. For robustness, use a proper JSON parser.
    weight_map_t* map = (weight_map_t*)calloc(1, sizeof(weight_map_t));
    map->capacity = 256;
    map->entries = (weight_entry_t*)calloc(map->capacity, sizeof(weight_entry_t));
    
    char line[1024];
    char current_name[256] = {0};
    int64_t current_offset = -1;
    int32_t current_shape[4] = {0};
    int current_num_dims = 0;
    int in_entry = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        // Look for layer name: "model.X.Y": {
        if (strstr(line, "\"")) {
            char* name_start = strchr(line, '"');
            if (name_start) {
                name_start++;
                char* name_end = strchr(name_start, '"');
                if (name_end) {
                    *name_end = '\0';
                    if (in_entry && current_offset >= 0) {
                        // Save previous entry
                        if (map->num_entries < map->capacity) {
                            weight_entry_t* e = &map->entries[map->num_entries++];
                            e->name = strdup(current_name);
                            e->offset = current_offset;
                            memcpy(e->shape, current_shape, sizeof(current_shape));
                            e->num_dims = current_num_dims;
                        }
                    }
                    strncpy(current_name, name_start, sizeof(current_name) - 1);
                    current_offset = -1;
                    current_num_dims = 0;
                    in_entry = 1;
                }
            }
        }
        
        // Look for "offset": value
        if (strstr(line, "\"offset\"")) {
            char* offset_str = strstr(line, ":");
            if (offset_str) {
                current_offset = (int64_t)strtoll(offset_str + 1, NULL, 10);
            }
        }
        
        // Look for "shape": [a, b, c, d]
        if (strstr(line, "\"shape\"")) {
            char* bracket = strchr(line, '[');
            if (bracket) {
                bracket++;
                char* token = strtok(bracket, ",]");
                current_num_dims = 0;
                while (token && current_num_dims < 4) {
                    current_shape[current_num_dims++] = (int32_t)strtol(token, NULL, 10);
                    token = strtok(NULL, ",]");
                }
            }
        }
    }
    
    // Save last entry
    if (in_entry && current_offset >= 0 && map->num_entries < map->capacity) {
        weight_entry_t* e = &map->entries[map->num_entries++];
        e->name = strdup(current_name);
        e->offset = current_offset;
        memcpy(e->shape, current_shape, sizeof(current_shape));
        e->num_dims = current_num_dims;
    }
    
    fclose(fp);
    return map;
}

static void free_weights_map(weight_map_t* map) {
    if (map) {
        for (int i = 0; i < map->num_entries; i++) {
            if (map->entries[i].name) {
                free(map->entries[i].name);
            }
        }
        free(map->entries);
        free(map);
    }
}

weights_loader_t* weights_loader_create(const char* weights_path) {
    if (!weights_path) return NULL;
    
    weights_loader_t* loader = (weights_loader_t*)calloc(1, sizeof(weights_loader_t));
    if (!loader) return NULL;
    
    // Load weights binary file
    FILE* fp = fopen(weights_path, "rb");
    if (!fp) {
        free(loader);
        return NULL;
    }
    
    // Get file size
    fseek(fp, 0, SEEK_END);
    loader->size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    // Allocate and read
    loader->data = (float*)malloc(loader->size);
    if (!loader->data) {
        fclose(fp);
        free(loader);
        return NULL;
    }
    
    if (fread(loader->data, 1, loader->size, fp) != loader->size) {
        free(loader->data);
        fclose(fp);
        free(loader);
        return NULL;
    }
    
    fclose(fp);
    
    // Try to find weights_map.json in same directory
    char map_path[1024];
    strncpy(map_path, weights_path, sizeof(map_path) - 1);
    char* last_slash = strrchr(map_path, '/');
    if (!last_slash) last_slash = strrchr(map_path, '\\');
    if (last_slash) {
        strcpy(last_slash + 1, "weights_map.json");
    } else {
        strcpy(map_path, "weights_map.json");
    }
    loader->map_path = strdup(map_path);
    
    return loader;
}

void weights_loader_free(weights_loader_t* loader) {
    if (loader) {
        if (loader->data) free(loader->data);
        if (loader->map_path) free((void*)loader->map_path);
        free(loader);
    }
}

float* weights_loader_get(weights_loader_t* loader, const char* name, 
                          int32_t* shape, int* num_dims) {
    if (!loader || !name) return NULL;
    
    // Parse weights_map.json
    weight_map_t* map = parse_weights_map(loader->map_path);
    if (!map) return NULL;
    
    // Find entry
    for (int i = 0; i < map->num_entries; i++) {
        if (strcmp(map->entries[i].name, name) == 0) {
            weight_entry_t* e = &map->entries[i];
            if (shape) {
                memcpy(shape, e->shape, sizeof(e->shape));
            }
            if (num_dims) {
                *num_dims = e->num_dims;
            }
            
            float* result = loader->data + (e->offset / sizeof(float));
            free_weights_map(map);
            return result;
        }
    }
    
    free_weights_map(map);
    return NULL;
}

int64_t weights_loader_get_offset(weights_loader_t* loader, const char* name) {
    if (!loader || !name) return -1;
    
    weight_map_t* map = parse_weights_map(loader->map_path);
    if (!map) return -1;
    
    for (int i = 0; i < map->num_entries; i++) {
        if (strcmp(map->entries[i].name, name) == 0) {
            int64_t offset = map->entries[i].offset;
            free_weights_map(map);
            return offset;
        }
    }
    
    free_weights_map(map);
    return -1;
}
