#ifndef DETECT_H
#define DETECT_H

#include "../core/tensor.h"

/**
 * Detection result (bounding box)
 */
typedef struct {
    float x;           // Center x (normalized 0-1)
    float y;           // Center y (normalized 0-1)
    float w;           // Width (normalized 0-1)
    float h;           // Height (normalized 0-1)
    float conf;        // Object confidence
    float cls_conf[80]; // Class confidences (80 classes for COCO)
    int32_t cls_id;    // Class ID (0-79)
} detection_t;

/**
 * Detect head parameters
 */
typedef struct {
    int32_t num_classes;      // 80 for COCO
    int32_t num_anchors;      // 3 per scale
    float anchors[3][6];      // Anchors for 3 scales [P3, P4, P5], each has 3 anchors (w, h)
    int32_t input_size;       // 640
} detect_params_t;

/**
 * Detect head output
 * Each scale outputs: (batch, 3, H, W, 85)
 * 85 = 4 (bbox) + 1 (obj_conf) + 80 (class_conf)
 */
typedef struct {
    tensor_t* p3_output;  // (1, 3, 80, 80, 85) → (1, 19200, 85)
    tensor_t* p4_output;  // (1, 3, 40, 40, 85) → (1, 4800, 85)
    tensor_t* p5_output;  // (1, 3, 20, 20, 85) → (1, 1200, 85)
} detect_output_t;

/**
 * Initialize detect parameters
 */
void detect_init_params(detect_params_t* params, int32_t num_classes, int32_t input_size);

/**
 * Forward pass: Apply 1×1 conv to P3, P4, P5 features
 * @param model Model instance (contains detect head conv layers)
 * @param p3_feature Input from layer 17 (1, 64, 80, 80) for YOLOv5n
 * @param p4_feature Input from layer 20 (1, 128, 40, 40) for YOLOv5n
 * @param p5_feature Input from layer 23 (1, 256, 20, 20) for YOLOv5n
 * @param output Output tensors (will be allocated)
 * @param params Detect parameters
 * @return 0 on success, -1 on failure
 */
int detect_forward(void* model, const tensor_t* p3_feature, const tensor_t* p4_feature, const tensor_t* p5_feature,
                   detect_output_t* output, const detect_params_t* params);

/**
 * Decode detection outputs to bounding boxes
 * @param output Detect output
 * @param params Detect parameters
 * @param detections Output array of detections (will be allocated)
 * @param num_detections Output number of detections
 * @param conf_threshold Confidence threshold (e.g., 0.25)
 * @return 0 on success, -1 on failure
 */
int detect_decode(const detect_output_t* output, const detect_params_t* params,
                  detection_t** detections, int32_t* num_detections, float conf_threshold);

/**
 * Free detect output
 */
void detect_free_output(detect_output_t* output);

/**
 * Free detections array
 */
void detect_free_detections(detection_t* detections, int32_t num_detections);

#endif // DETECT_H
