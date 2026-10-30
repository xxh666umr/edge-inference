/* 自动生成，请勿手改：tools/train_modrec.py -> tools/quantize.py */
#ifndef MODEL_H
#define MODEL_H
#include <stdint.h>

#define MODEL_NAME "radio_ml_cnn_2016"
#define MODEL_INPUT_C     2
#define MODEL_INPUT_LEN   128
#define MODEL_NUM_CLASSES 11
#define MODEL_INPUT_SCALE 0.041700198f
#define MODEL_PARAM_COUNT 3195
#define MODEL_CONV1_FX_MUL   39325
#define MODEL_CONV1_FX_SHIFT 24
#define MODEL_CONV2_FX_MUL   146814
#define MODEL_CONV2_FX_SHIFT 24

#ifdef __cplusplus
extern "C" {
#endif

/* 输入: int8[MODEL_INPUT_C][MODEL_INPUT_LEN] 行主序（I 通道在前）。
 * 返回 argmax 类别，probs 填 MODEL_NUM_CLASSES 个概率。 */
int model_forward(const int8_t *input, float *probs);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
