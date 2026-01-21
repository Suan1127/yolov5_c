#include "yolov5n_graph.h"

/**
 * YOLOv5n model graph
 * 
 * Backbone (0-9):
 *   0: Conv(3→16, 6×6, s=2, p=2)     [input] for YOLOv5n
 *   1: Conv(16→32, 3×3, s=2)          [0] for YOLOv5n
 *   2: C3(32→32, n=1)                 [1] for YOLOv5n
 *   3: Conv(32→64, 3×3, s=2)         [2] → save[3] for YOLOv5n
 *   4: C3(64→64, n=2)               [3] → save[4] for YOLOv5n
 *   5: Conv(64→128, 3×3, s=2)        [4] → save[5] for YOLOv5n
 *   6: C3(128→128, n=3)               [5] → save[6] for YOLOv5n
 *   7: Conv(128→256, 3×3, s=2)        [6] → save[7] for YOLOv5n
 *   8: C3(256→256, n=1)               [7] for YOLOv5n
 *   9: SPPF(256→256, k=5)             [8] → save[9] for YOLOv5n
 *
 * Head (10-23):
 *   10: Conv(256→128, 1×1)            [9] for YOLOv5n
 *   11: Upsample(×2)                  [10]
 *   12: Concat([11, 6])               [11, 6] → 256 channels for YOLOv5n
 *   13: C3(256→128, n=1, shortcut=False) [12] for YOLOv5n
 *   14: Conv(128→64, 1×1)            [13] for YOLOv5n
 *   15: Upsample(×2)                  [14]
 *   16: Concat([15, 4])               [15, 4] → 128 channels for YOLOv5n
 *   17: C3(128→64, n=1, shortcut=False) [16] → save[17] (P3) for YOLOv5n
 *   18: Conv(64→64, 3×3, s=2)       [17] for YOLOv5n
 *   19: Concat([18, 14])              [18, 14] → 128 channels for YOLOv5n
 *   20: C3(128→128, n=1, shortcut=False) [19] → save[20] (P4) for YOLOv5n
 *   21: Conv(128→128, 3×3, s=2)       [20] for YOLOv5n
 *   22: Concat([21, 10])              [21, 10] → 256 channels for YOLOv5n
 *   23: C3(256→256, n=1, shortcut=False) [22] → save[23] (P5) for YOLOv5n
 *   24: Detect([17, 20, 23])           [17, 20, 23]
 */
const layer_def_t yolov5n_graph[YOLOV5N_NUM_LAYERS] = {
    // Backbone
    {0, LAYER_CONV, {1, {0, 0, 0, 0}}, 0},                    // 0: Conv [input]
    {1, LAYER_CONV, {1, {0, 0, 0, 0}}, 0},                    // 1: Conv [0]
    {2, LAYER_C3, {1, {1, 0, 0, 0}}, 0},                      // 2: C3 [1]
    {3, LAYER_CONV, {1, {2, 0, 0, 0}}, 1},                    // 3: Conv [2] → save
    {4, LAYER_C3, {1, {3, 0, 0, 0}}, 1},                      // 4: C3 [3] → save
    {5, LAYER_CONV, {1, {4, 0, 0, 0}}, 1},                    // 5: Conv [4] → save
    {6, LAYER_C3, {1, {5, 0, 0, 0}}, 1},                      // 6: C3 [5] → save
    {7, LAYER_CONV, {1, {6, 0, 0, 0}}, 1},                    // 7: Conv [6] → save
    {8, LAYER_C3, {1, {7, 0, 0, 0}}, 0},                      // 8: C3 [7]
    {9, LAYER_SPPF, {1, {8, 0, 0, 0}}, 1},                    // 9: SPPF [8] → save
    
    // Head
    {10, LAYER_CONV, {1, {9, 0, 0, 0}}, 0},                   // 10: Conv [9]
    {11, LAYER_UPSAMPLE, {1, {10, 0, 0, 0}}, 0},             // 11: Upsample [10]
    {12, LAYER_CONCAT, {2, {11, 6, 0, 0}}, 0},                // 12: Concat [11, 6]
    {13, LAYER_C3, {1, {12, 0, 0, 0}}, 0},                    // 13: C3 [12]
    {14, LAYER_CONV, {1, {13, 0, 0, 0}}, 0},                  // 14: Conv [13]
    {15, LAYER_UPSAMPLE, {1, {14, 0, 0, 0}}, 0},             // 15: Upsample [14]
    {16, LAYER_CONCAT, {2, {15, 4, 0, 0}}, 0},                // 16: Concat [15, 4]
    {17, LAYER_C3, {1, {16, 0, 0, 0}}, 1},                    // 17: C3 [16] → save (P3)
    {18, LAYER_CONV, {1, {17, 0, 0, 0}}, 0},                  // 18: Conv [17]
    {19, LAYER_CONCAT, {2, {18, 13, 0, 0}}, 0},               // 19: Concat [18, 13]
    {20, LAYER_C3, {1, {19, 0, 0, 0}}, 1},                    // 20: C3 [19] → save (P4)
    {21, LAYER_CONV, {1, {20, 0, 0, 0}}, 0},                  // 21: Conv [20]
    {22, LAYER_CONCAT, {2, {21, 10, 0, 0}}, 0},               // 22: Concat [21, 10]
    {23, LAYER_C3, {1, {22, 0, 0, 0}}, 1},                    // 23: C3 [22] → save (P5)
    {24, LAYER_DETECT, {3, {17, 20, 23, 0}}, 0},              // 24: Detect [17, 20, 23]
};

const layer_def_t* yolov5n_get_layer(int32_t index) {
    if (index < 0 || index >= YOLOV5N_NUM_LAYERS) {
        return NULL;
    }
    return &yolov5n_graph[index];
}

void yolov5n_get_save_list(int32_t* save_list, int32_t* num_saves) {
    if (!save_list || !num_saves) return;
    
    *num_saves = 0;
    for (int i = 0; i < YOLOV5N_NUM_LAYERS; i++) {
        if (yolov5n_graph[i].save) {
            save_list[(*num_saves)++] = i;
        }
    }
}
