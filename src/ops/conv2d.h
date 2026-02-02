#ifndef CONV2D_H
#define CONV2D_H

#include "../core/tensor.h"

/**
 * 2D Convolution parameters
 */
typedef struct {
    int32_t out_channels;
    int32_t kernel_size;   // Assuming square kernel (k×k)
    int32_t stride;
    int32_t padding;
    int32_t groups;        // For depthwise conv (default: 1)
    int32_t dilation;      // Default: 1
} conv2d_params_t;

/**
 * Convolution layer with weights and bias
 */
typedef struct {
    conv2d_params_t params;
    float* weight;         // [out_channels, in_channels, k, k]
    float* bias;           // [out_channels] or NULL
    int32_t in_channels;
} conv2d_layer_t;

/**
 * Design: Separation of Responsibilities (HW acceleration)
 * - conv2d_forward / conv2d_fused_bn_silu_forward do: dimension validation,
 *   loop over (b, oc, oh, ow), then invoke a one-pixel kernel and write result.
 * - One-pixel kernels (internal in conv2d.c): conv1x1_one_pixel(b, oc, h, w),
 *   conv3x3_one_pixel(b, oc, oh, ow), conv_generic_one_pixel(b, oc, oh, ow).
 *   Each computes a single output pixel (MAC only); short combinational path.
 * - HW: Replace the one_pixel implementations with a custom accelerator or
 *   platform-specific conv2d.c for incremental HW/SW co-design.
 */

/**
 * Initialize convolution layer
 */
int conv2d_init(conv2d_layer_t* layer, int32_t in_channels, const conv2d_params_t* params);

/**
 * Free convolution layer
 */
void conv2d_free(conv2d_layer_t* layer);

/**
 * Forward pass: output = conv2d(input)
 */
int conv2d_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output);

/**
 * Fused Conv+BN+SiLU: Conv 결과를 DDR에 쓰지 않고 레지스터에서 BN+SiLU 후 한 번만 기록.
 * DDR 읽기/쓰기 감소 (임베디드 메모리 최적화). bn NULL이면 Conv+SiLU만 (fused BN인 경우).
 */
#ifndef CONV2D_FUSED_FWD_DECL
struct batchnorm2d_layer_t;
#endif
int conv2d_fused_bn_silu_forward(const conv2d_layer_t* layer, const struct batchnorm2d_layer_t* bn,
                                  const tensor_t* input, tensor_t* output);

/**
 * Golden 디버그: 한 점(b,oc,oh,ow)에 대해 conv_raw / bn_out / silu_out 을 hex로 출력.
 * 임베디드에서 같은 점(예: Layer0 첫 float b=0,oc=0,oh=0,ow=0)으로 비교용.
 */
void conv2d_fused_debug_one_pixel(const conv2d_layer_t* layer, const struct batchnorm2d_layer_t* bn,
                                   const tensor_t* input, int32_t b, int32_t oc, int32_t oh, int32_t ow);

/**
 * Golden 상세 덤프: bias[oc], acc_after_ic0/1/2, w[oc][0] 앞 8개, x[b][0] (0,0)~(1,1) (임베디드 비교용).
 */
void conv2d_debug_one_pixel_detail(const conv2d_layer_t* layer, const tensor_t* input,
                                   int32_t b, int32_t oc, int32_t oh, int32_t ow);

/**
 * Load weights from buffer
 * @param layer Layer to load weights into
 * @param weight_buf Weight buffer [out_c, in_c, k, k]
 * @param bias_buf Bias buffer [out_c] or NULL
 */
int conv2d_load_weights(conv2d_layer_t* layer, const float* weight_buf, const float* bias_buf);

#endif // CONV2D_H
