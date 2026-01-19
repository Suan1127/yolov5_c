#ifndef YOLOV5S_GRAPH_H
#define YOLOV5S_GRAPH_H

#include <stdint.h>

/**
 * Layer types
 */
typedef enum {
    LAYER_CONV,
    LAYER_C3,
    LAYER_SPPF,
    LAYER_UPSAMPLE,
    LAYER_CONCAT,
    LAYER_DETECT
} layer_type_t;

/**
 * Layer connection (which layers feed into this layer)
 */
typedef struct {
    int32_t num_inputs;
    int32_t inputs[4];  // Max 4 inputs (for Concat)
} layer_connection_t;

/**
 * Layer definition
 */
typedef struct {
    int32_t index;
    layer_type_t type;
    layer_connection_t connection;
    int32_t save;  // 1 if this layer output should be saved for later use
} layer_def_t;

/**
 * YOLOv5s model graph definition
 * Based on PROJECT_BRIEF.md
 */
#define YOLOV5S_NUM_LAYERS 25

extern const layer_def_t yolov5s_graph[YOLOV5S_NUM_LAYERS];

/**
 * Get layer definition by index
 */
const layer_def_t* yolov5s_get_layer(int32_t index);

/**
 * Get save list (layers whose outputs should be saved)
 */
void yolov5s_get_save_list(int32_t* save_list, int32_t* num_saves);

#endif // YOLOV5S_GRAPH_H
