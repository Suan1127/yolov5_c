#include "weights_loader.h"
#include "../../third_party/jsmn/jsmn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define strdup _strdup
#endif

// Weight entry structure
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

// Helper: Read entire file into buffer
static char* read_file_to_buffer(const char* path, size_t* out_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    
    size_t read = fread(buffer, 1, file_size, fp);
    fclose(fp);
    
    if (read != (size_t)file_size) {
        free(buffer);
        return NULL;
    }
    
    buffer[file_size] = '\0';
    if (out_size) *out_size = file_size;
    return buffer;
}

// Helper: Extract string from token
static char* extract_string(const char* json, const jsmntok_t* tok) {
    int len = tok->end - tok->start;
    char* str = (char*)malloc(len + 1);
    if (!str) return NULL;
    memcpy(str, json + tok->start, len);
    str[len] = '\0';
    return str;
}

// Helper: Skip a token and all its children (for nested objects/arrays)
static int skip_token(const jsmntok_t* tokens, int num_tokens, int idx) {
    if (idx >= num_tokens) return idx;
    
    const jsmntok_t* tok = &tokens[idx];
    int size = tok->size;
    
    if (tok->type == JSMN_OBJECT) {
        // For objects: size is number of key-value pairs
        // Each pair = 1 key (STRING) + 1 value (any type)
        idx++;  // Skip the object token itself
        for (int i = 0; i < size && idx < num_tokens; i++) {
            // Skip key (STRING)
            idx = skip_token(tokens, num_tokens, idx);
            // Skip value (any type)
            if (idx < num_tokens) {
                idx = skip_token(tokens, num_tokens, idx);
            }
        }
    } else if (tok->type == JSMN_ARRAY) {
        // For arrays: size is number of elements
        idx++;  // Skip the array token itself
        for (int i = 0; i < size && idx < num_tokens; i++) {
            idx = skip_token(tokens, num_tokens, idx);
        }
    } else {
        // For primitives/strings, just skip one token
        idx++;
    }
    
    return idx;
}

// Helper: Find value for a key in an object
static int find_key_value(const char* json, const jsmntok_t* tokens, int num_tokens,
                         int obj_start, const char* key, jsmntok_t* out_value) {
    // obj_start points to the object token
    if (obj_start >= num_tokens || tokens[obj_start].type != JSMN_OBJECT) {
        return -1;
    }
    
    int obj_size = tokens[obj_start].size;
    int i = obj_start + 1;  // Start after the object token
    
    // Iterate through key-value pairs
    for (int pair = 0; pair < obj_size && i < num_tokens; pair++) {
        // Key token
        jsmntok_t* key_tok = (jsmntok_t*)&tokens[i];
        if (key_tok->type != JSMN_STRING) {
            i = skip_token(tokens, num_tokens, i);
            continue;
        }
        
        // Check if this key matches
        int key_len = key_tok->end - key_tok->start;
        if (key_len > 0 && strncmp(json + key_tok->start, key, key_len) == 0 && 
            (key[key_len] == '\0' || key_len == (int)strlen(key))) {
            // Found the key, next token is the value
            i++;
            if (i < num_tokens) {
                *out_value = tokens[i];
                return i;  // Return index of value token
            }
        } else {
            // Skip key and value tokens
            i++;  // Skip key
            if (i < num_tokens) {
                i = skip_token(tokens, num_tokens, i);  // Skip value
            }
        }
    }
    
    return -1;  // Key not found
}

// Parse weights_map.json using jsmn
static weight_map_t* parse_weights_map(const char* map_path) {
    // Debug output removed - only show errors
    
    // Read entire file
    size_t file_size;
    char* json = read_file_to_buffer(map_path, &file_size);
    if (!json) {
        fprintf(stderr, "parse_weights_map: Failed to read file\n");
        return NULL;
    }
    
    // Initialize parser
    jsmn_parser parser;
    jsmn_init(&parser);
    
    // First pass: count tokens (pass NULL to get count)
    int num_tokens = jsmn_parse(&parser, json, file_size, NULL, 0);
    if (num_tokens < 0) {
        fprintf(stderr, "parse_weights_map: JSON parse error: %d\n", num_tokens);
        free(json);
        return NULL;
    }
    
    // Allocate tokens
    jsmntok_t* tokens = (jsmntok_t*)malloc(sizeof(jsmntok_t) * num_tokens);
    if (!tokens) {
        fprintf(stderr, "parse_weights_map: Failed to allocate tokens\n");
        free(json);
        return NULL;
    }
    
    // Second pass: actual parsing
    jsmn_init(&parser);
    int r = jsmn_parse(&parser, json, file_size, tokens, num_tokens);
    if (r < 0) {
        fprintf(stderr, "parse_weights_map: JSON parse error: %d\n", r);
        free(tokens);
        free(json);
        return NULL;
    }
    
    // Create weight map
    weight_map_t* map = (weight_map_t*)calloc(1, sizeof(weight_map_t));
    map->capacity = 256;
    map->entries = (weight_entry_t*)calloc(map->capacity, sizeof(weight_entry_t));
    
    // Root should be an object
    if (r == 0 || tokens[0].type != JSMN_OBJECT) {
        fprintf(stderr, "parse_weights_map: Root is not an object\n");
        free(tokens);
        free(json);
        free(map->entries);
        free(map);
        return NULL;
    }
    
    // Parse each entry in the root object
    int root_size = tokens[0].size;
    int i = 1;  // Start after root object token
    int entry_count = 0;
    
    for (int pair = 0; pair < root_size && i < r; pair++) {
        // Key (entry name)
        if (i >= r || tokens[i].type != JSMN_STRING) {
            // Skip non-string token
            i = skip_token(tokens, r, i);
            continue;
        }
        
        char* entry_name = extract_string(json, &tokens[i++]);
        if (!entry_name) {
            // If extraction failed, skip to next token
            if (i < r) i = skip_token(tokens, r, i);
            continue;
        }
        
        // Value (entry object)
        if (i >= r || tokens[i].type != JSMN_OBJECT) {
            free(entry_name);
            // Skip non-object value
            if (i < r) i = skip_token(tokens, r, i);
            continue;
        }
        
        int entry_obj_idx = i;
        i++;  // Move past the object token
        const jsmntok_t* entry_obj = &tokens[entry_obj_idx];
        
        // Find "offset" key
        jsmntok_t offset_tok;
        int offset_idx = find_key_value(json, tokens, r, entry_obj_idx, "offset", &offset_tok);
        
        // Find "shape" key
        jsmntok_t shape_tok;
        int shape_idx = find_key_value(json, tokens, r, entry_obj_idx, "shape", &shape_tok);
        
        if (offset_idx >= 0 && offset_tok.type == JSMN_PRIMITIVE) {
            // Parse offset value
            char offset_str[64];
            int offset_len = offset_tok.end - offset_tok.start;
            if (offset_len < (int)sizeof(offset_str)) {
                memcpy(offset_str, json + offset_tok.start, offset_len);
                offset_str[offset_len] = '\0';
                int64_t offset = (int64_t)strtoll(offset_str, NULL, 10);
                
                // Parse shape array
                int32_t shape[4] = {0};
                int num_dims = 0;
                
                if (shape_idx >= 0 && shape_tok.type == JSMN_ARRAY) {
                    int shape_size = shape_tok.size;
                    int shape_i = shape_idx + 1;  // Start after array token
                    
                    for (int dim = 0; dim < shape_size && dim < 4 && shape_i < r; dim++) {
                        const jsmntok_t* dim_tok = &tokens[shape_i];
                        if (dim_tok->type == JSMN_PRIMITIVE) {
                            char dim_str[32];
                            int dim_len = dim_tok->end - dim_tok->start;
                            if (dim_len < (int)sizeof(dim_str) && dim_len > 0) {
                                memcpy(dim_str, json + dim_tok->start, dim_len);
                                dim_str[dim_len] = '\0';
                                shape[dim] = (int32_t)strtol(dim_str, NULL, 10);
                                num_dims++;
                            }
                            // Move to next element (primitive is just one token)
                            shape_i++;
                        } else {
                            // Skip non-primitive elements (shouldn't happen in shape arrays)
                            shape_i = skip_token(tokens, r, shape_i);
                        }
                    }
                }
                
                // Save entry
                if (map->num_entries < map->capacity) {
                    weight_entry_t* e = &map->entries[map->num_entries++];
                    e->name = entry_name;
                    e->offset = offset;
                    memcpy(e->shape, shape, sizeof(shape));
                    e->num_dims = num_dims;
                    
                    entry_count++;
                } else {
                    free(entry_name);
                }
            } else {
                free(entry_name);
            }
        } else {
            free(entry_name);
        }
        
        // Skip to next entry (skip all tokens in this entry object)
        i = skip_token(tokens, r, entry_obj_idx);
    }
    
    free(tokens);
    free(json);
    
    // Only show warning if parsing failed
    if (map->num_entries == 0) {
        fprintf(stderr, "WARNING: parse_weights_map: No entries parsed!\n");
    }
    
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
    if (!loader || !name) {
        fprintf(stderr, "weights_loader_get: NULL loader or name\n");
        return NULL;
    }
    
    // Parse weights_map.json
    weight_map_t* map = parse_weights_map(loader->map_path);
    if (!map) {
        fprintf(stderr, "Error: weights_loader_get: Failed to parse weights_map.json from %s\n", loader->map_path);
        return NULL;
    }
    
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
    
    // Don't print "not found" message - it's normal for optional weights (e.g., BN weights when fused)
    // Only print error for actual failures (handled by caller)
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
