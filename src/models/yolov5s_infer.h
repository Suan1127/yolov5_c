#ifndef YOLOV5S_INFER_H
#define YOLOV5S_INFER_H

#include "yolov5s_build.h"
#include "../core/tensor.h"

/**
 * Forward pass through YOLOv5s model
 * @param model Model instance
 * @param input Input tensor (1, 3, 640, 640)
 * @param output Output tensors for Detect head [P3, P4, P5]
 *   - output[0]: P3 (1, 128, 80, 80)
 *   - output[1]: P4 (1, 256, 40, 40)
 *   - output[2]: P5 (1, 512, 20, 20)
 * @return 0 on success, -1 on failure
 */
int yolov5s_forward(yolov5s_model_t* model, const tensor_t* input, tensor_t* output[3]);

/**
 * Get saved feature map by layer index
 * @param model Model instance
 * @param layer_idx Layer index (3, 4, 5, 6, 7, 9, 17, 20, 23)
 * @return Saved feature tensor or NULL if not found
 */
tensor_t* yolov5s_get_saved_feature(yolov5s_model_t* model, int32_t layer_idx);

/**
 * Get P3, P4, P5 features for Detect head
 * @param model Model instance
 * @param p3 Output P3 feature (will be set to saved feature)
 * @param p4 Output P4 feature (will be set to saved feature)
 * @param p5 Output P5 feature (will be set to saved feature)
 * @return 0 on success, -1 on failure
 */
int yolov5s_get_detect_features(yolov5s_model_t* model, tensor_t** p3, tensor_t** p4, tensor_t** p5);

#endif // YOLOV5S_INFER_H
