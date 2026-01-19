#ifndef NMS_H
#define NMS_H

#include "detect.h"

/**
 * Non-Maximum Suppression
 * Removes overlapping detections, keeping only the highest confidence ones
 * 
 * @param detections Input detections array
 * @param num_detections Number of input detections
 * @param output_detections Output array (will be allocated)
 * @param output_count Output number of detections
 * @param iou_threshold IoU threshold (e.g., 0.45)
 * @param max_detections Maximum number of detections to keep
 * @return 0 on success, -1 on failure
 */
int nms(detection_t* detections, int32_t num_detections,
        detection_t** output_detections, int32_t* output_count,
        float iou_threshold, int32_t max_detections);

/**
 * Calculate IoU (Intersection over Union) between two boxes
 */
float calculate_iou(const detection_t* box1, const detection_t* box2);

#endif // NMS_H
