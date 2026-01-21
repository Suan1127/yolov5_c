#ifndef YOLOV5N_INFER_H
#define YOLOV5N_INFER_H

#include "yolov5n_build.h"
#include "../core/tensor.h"

/**
 * Forward pass through YOLOv5n model
 * @param model Model instance
 * @param input Input tensor (1, 3, 640, 640)
 * @param output Output tensors for Detect head [P3, P4, P5]
 *   - output[0]: P3 (1, 64, 80, 80) for YOLOv5n
 *   - output[1]: P4 (1, 128, 40, 40) for YOLOv5n
 *   - output[2]: P5 (1, 256, 20, 20) for YOLOv5n
 * @return 0 on success, -1 on failure
 */
int yolov5n_forward(yolov5n_model_t* model, const tensor_t* input, tensor_t* output[3]);

/**
 * Get saved feature map by layer index
 * @param model Model instance
 * @param layer_idx Layer index (3, 4, 5, 6, 7, 9, 17, 20, 23)
 * @return Saved feature tensor or NULL if not found
 */
tensor_t* yolov5n_get_saved_feature(yolov5n_model_t* model, int32_t layer_idx);

/**
 * Get P3, P4, P5 features for Detect head
 * @param model Model instance
 * @param p3 Output P3 feature (will be set to saved feature)
 * @param p4 Output P4 feature (will be set to saved feature)
 * @param p5 Output P5 feature (will be set to saved feature)
 * @return 0 on success, -1 on failure
 */
int yolov5n_get_detect_features(yolov5n_model_t* model, tensor_t** p3, tensor_t** p4, tensor_t** p5);

/**
 * Set output directory for saving intermediate layer outputs
 * @param model Model instance
 * @param output_dir Directory path (NULL to disable saving)
 * @return 0 on success, -1 on failure
 */
int yolov5n_set_output_dir(yolov5n_model_t* model, const char* output_dir);

/**
 * Save all saved features to files
 * @param model Model instance
 * @return 0 on success, -1 on failure
 */
int yolov5n_save_features(yolov5n_model_t* model);

#endif // YOLOV5N_INFER_H
