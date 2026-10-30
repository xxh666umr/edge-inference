#ifndef NN_H
#define NN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 全连接层累加：acc[o] = bias[o] + sum_i W[o*in_dim+i] * x[i]，int32 累加 */
void matmul_accumulate(const int8_t *w, const int32_t *bias, const int8_t *x,
                       int32_t *acc, int in_dim, int out_dim);

/* 重新量化并带 ReLU：y[i] = clamp(round(acc[i]*req_scale), 0, 127) */
void requantize_relu(const int32_t *acc, int8_t *y, int n, float req_scale);

/* 对 int32 logits 做 softmax，返回最大概率类别 */
int softmax_argmax(const int32_t *logits, int n, float acc_scale, float *probs);

/* 一维卷积累加（valid padding，不做量化/激活）：
 *   x    布局 [c_in][l_in]，行主序（先通道、后长度）；
 *   w    布局 [c_out][c_in][k]；
 *   bias 布局 [c_out]，可为 NULL（表示偏置全 0）；
 *   acc  布局 [c_out][l_out]，l_out = (l_in - k) / stride + 1（要求 l_in - k 能被 stride 整除）。
 *   acc[o][t] = bias[o] + sum_{c< c_in} sum_{j< k} w[o][c][j] * x[c][t*stride + j]
 * 全部为整数运算，int32 累加，无动态内存。 */
void conv1d_accumulate(const int8_t *w, const int32_t *bias, const int8_t *x,
                       int32_t *acc, int c_in, int c_out, int l_in, int k, int stride);

/* 通用重新量化（不含 ReLU）：
 *   y[i] = clamp(round(acc[i] * scale), -128, 127)
 * 其中 round 为「四舍五入、远离零」（round half away from zero）。 */
void requantize(const int32_t *acc, int8_t *y, int n, float scale);

/* 一维最大池化：x / y 布局均为 [c][l]，
 *   y[c][t] = max_{j< k} x[c][t*stride + j]
 *   l_out = (l_in - k) / stride + 1（要求 l_in - k 能被 stride 整除）。 */
void maxpool1d(const int8_t *x, int8_t *y, int c, int l_in, int k, int stride);

/* 一维平均池化：x / y 布局均为 [c][l]，
 *   y[c][t] = round(mean_{j< k} x[c][t*stride + j])
 * 取整方式与 requantize 一致：四舍五入、远离零；结果再夹到 int8 范围。 */
void avgpool1d(const int8_t *x, int8_t *y, int c, int l_in, int k, int stride);

/* ---------- 定点重量化（F103 无 FPU，替代 float scale，快 ~20 倍） ----------
 * mul / shift 由导出器从 float scale 近似得到：scale ≈ mul / 2^shift。
 * 计算：p = (int64_t)acc[i] * mul + (1 << (shift-1))；r = p >> shift。
 * 说明：对 ReLU 输出，负值最终会被夹到 0，故负数的向下取整不影响结果；
 *       正值等价于 round-half-up，与 float 实现的 round-half-away 在非负域一致。
 * 要求 0 < shift <= 30，mul > 0；中间结果用 int64，不会溢出。 */
void requantize_relu_fx(const int32_t *acc, int8_t *y, int n, int32_t mul, int32_t shift);
void requantize_fx(const int32_t *acc, int8_t *y, int n, int32_t mul, int32_t shift);

/* 普通 softmax（不返回 argmax）：probs[i] = exp(logits[i]*acc_scale - max) / sum */
void softmax_probs(const int32_t *logits, int n, float acc_scale, float *probs);

/* ---------- 序列模型算子（float32 参考实现） ----------
 * 说明：F103 无 FPU，这些算子用软浮点较慢（适合 M4F/H7 或小规模模型）；
 *       接口与 PyTorch 语义一致，便于换模型后直接验证。
 * 本阶段用于把推理库从“CNN 专用”扩展到 RNN / Transformer 组件，
 * 正确性由 tests/test_seq_ops_parity.py 与 numpy 对齐验证。 */

/* 单步 GRU（gate 顺序 r, z, n，与 PyTorch nn.GRUCell 一致）：
 *   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *   n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
 *   h_new = (1 - z) * n + z * h
 * w_ih 布局 [3H][input_dim]，w_hh 布局 [3H][hidden]，b_* 布局 [3H]。 */
void gru_cell_f32(const float *w_ih, const float *w_hh,
                  const float *b_ih, const float *b_hh,
                  const float *x, const float *h, float *h_new,
                  int input_dim, int hidden);

/* LayerNorm：y = (x - mean) / sqrt(var + eps) * gamma + beta（var 为有偏方差） */
void layernorm_f32(const float *x, const float *gamma, const float *beta,
                   float *y, int n, float eps);

/* 单头 scaled dot-product attention：
 *   scores = Q K^T / sqrt(d_model)，按行 softmax，out = softmax(scores) V
 * q/k/v 布局 [seq_len][d_model]，scores 为调用者提供的 [seq_len][seq_len] 缓冲，
 * out 布局 [seq_len][d_model]。 */
void attention_f32(const float *q, const float *k, const float *v, float *out,
                   int seq_len, int d_model, float *scores);

#ifdef __cplusplus
}
#endif

#endif /* NN_H */
