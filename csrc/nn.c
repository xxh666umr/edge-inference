#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* 把热循环函数放到 .ramfunc 段（链接脚本 VMA=RAM / LMA=Flash，启动时拷贝），
 * Cortex-M3 从 SRAM 取指无 Flash 等待周期；PC 端（MSVC）编译时该属性为空。 */
#if defined(__arm__)
#define RAMFUNC __attribute__((section(".ramfunc"), noinline))
#else
#define RAMFUNC
#endif

RAMFUNC void matmul_accumulate(const int8_t *w, const int32_t *bias, const int8_t *x,
                       int32_t *acc, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        int32_t s = bias ? bias[o] : 0;
        const int8_t *wrow = w + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            s += (int32_t)wrow[i] * (int32_t)x[i];
        }
        acc[o] = s;
    }
}

void requantize_relu(const int32_t *acc, int8_t *y, int n, float req_scale) {
    for (int i = 0; i < n; ++i) {
        float v = (float)acc[i] * req_scale;
        int32_t r = (int32_t)(v + (v >= 0 ? 0.5f : -0.5f));
        if (r < 0) r = 0;
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        y[i] = (int8_t)r;
    }
}

int softmax_argmax(const int32_t *logits, int n, float acc_scale, float *probs) {
    float mx = (float)logits[0] * acc_scale;
    for (int i = 1; i < n; ++i) {
        float v = (float)logits[i] * acc_scale;
        if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (float)exp((float)logits[i] * acc_scale - mx);
        sum += probs[i];
    }
    int arg = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (float)(probs[i] / sum);
        if (probs[i] > probs[arg]) arg = i;
    }
    return arg;
}

/* ============================ 新增算子 ============================ */

/* 把浮点值四舍五入为整数，舍入方向为「远离零」：
 *   0.5 -> 1, 1.4 -> 1, -0.5 -> -1, -1.4 -> -1
 * 与 requantize_relu 中的取整写法保持一致（float32 运算，C 的 int 转换向零截断）。 */
static int32_t round_away_from_zero(float v) {
    return (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
}

RAMFUNC void conv1d_accumulate(const int8_t *w, const int32_t *bias, const int8_t *x,
                       int32_t *acc, int c_in, int c_out, int l_in, int k, int stride) {
    const int l_out = (l_in - k) / stride + 1;
    for (int o = 0; o < c_out; ++o) {
        /* w 的布局是 [c_out][c_in][k]，第 o 个输出通道对应连续的一段 c_in*k 个权重 */
        const int8_t *wo = w + (size_t)o * (size_t)c_in * (size_t)k;
        for (int t = 0; t < l_out; ++t) {
            int32_t s = bias ? bias[o] : 0;
            const int base = t * stride; /* 该输出位置在输入上的起点（valid padding） */
            for (int c = 0; c < c_in; ++c) {
                const int8_t *xc = x + (size_t)c * (size_t)l_in;
                const int8_t *wc = wo + (size_t)c * (size_t)k;
                for (int j = 0; j < k; ++j) {
                    s += (int32_t)wc[j] * (int32_t)xc[base + j];
                }
            }
            acc[(size_t)o * (size_t)l_out + (size_t)t] = s;
        }
    }
}

void requantize(const int32_t *acc, int8_t *y, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        int32_t r = round_away_from_zero((float)acc[i] * scale);
        if (r < -128) r = -128;
        if (r > 127) r = 127;
        y[i] = (int8_t)r;
    }
}

RAMFUNC void maxpool1d(const int8_t *x, int8_t *y, int c, int l_in, int k, int stride) {
    const int l_out = (l_in - k) / stride + 1;
    for (int ch = 0; ch < c; ++ch) {
        const int8_t *xc = x + (size_t)ch * (size_t)l_in;
        int8_t *yc = y + (size_t)ch * (size_t)l_out;
        for (int t = 0; t < l_out; ++t) {
            const int base = t * stride;
            int8_t m = xc[base];
            for (int j = 1; j < k; ++j) {
                int8_t v = xc[base + j];
                if (v > m) m = v;
            }
            yc[t] = m;
        }
    }
}

RAMFUNC void avgpool1d(const int8_t *x, int8_t *y, int c, int l_in, int k, int stride) {
    const int l_out = (l_in - k) / stride + 1;
    for (int ch = 0; ch < c; ++ch) {
        const int8_t *xc = x + (size_t)ch * (size_t)l_in;
        int8_t *yc = y + (size_t)ch * (size_t)l_out;
        for (int t = 0; t < l_out; ++t) {
            const int base = t * stride;
            int32_t sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += (int32_t)xc[base + j];
            }
            /* 先求和再除，用 float32 求均值后「四舍五入、远离零」；k 很小、和也在 int32 内，
             * 因此与测试里的 numpy 参考实现逐位一致。 */
            float mean = (float)sum / (float)k;
            int32_t r = round_away_from_zero(mean);
            if (r < -128) r = -128;
            if (r > 127) r = 127;
            yc[t] = (int8_t)r;
        }
    }
}

/* ======================= 定点重量化（int64 中间结果） =======================
 * 与 requantize_relu 的 float 语义等价（非负域），但只用整数乘加/移位，
 * 在 Cortex-M3（无 FPU）上比软浮点快一个数量级。 */
RAMFUNC void requantize_relu_fx(const int32_t *acc, int8_t *y, int n, int32_t mul, int32_t shift) {
    const int64_t half = (int64_t)1 << (shift - 1);
    for (int i = 0; i < n; ++i) {
        int64_t p = (int64_t)acc[i] * (int64_t)mul + half;
        int32_t r = (int32_t)(p >> shift);   /* 算术右移；负数向下取整后被夹到 0 */
        if (r < 0) r = 0;
        if (r > 127) r = 127;
        y[i] = (int8_t)r;
    }
}

RAMFUNC void requantize_fx(const int32_t *acc, int8_t *y, int n, int32_t mul, int32_t shift) {
    const int64_t half = (int64_t)1 << (shift - 1);
    for (int i = 0; i < n; ++i) {
        int64_t p = (int64_t)acc[i] * (int64_t)mul + half;
        int32_t r = (int32_t)(p >> shift);
        if (r < -128) r = -128;
        if (r > 127) r = 127;
        y[i] = (int8_t)r;
    }
}

/* ========================= 序列模型算子（float32） ========================= */

void softmax_probs(const int32_t *logits, int n, float acc_scale, float *probs) {
    float mx = (float)logits[0] * acc_scale;
    for (int i = 1; i < n; ++i) {
        float v = (float)logits[i] * acc_scale;
        if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = (float)exp((float)logits[i] * acc_scale - mx);
        sum += probs[i];
    }
    for (int i = 0; i < n; ++i) {
        probs[i] = (float)(probs[i] / sum);
    }
}


void gru_cell_f32(const float *w_ih, const float *w_hh,
                  const float *b_ih, const float *b_hh,
                  const float *x, const float *h, float *h_new,
                  int input_dim, int hidden) {
    /* PyTorch 布局：w_ih [3H][input_dim]，w_hh [3H][hidden]，行顺序 r, z, n。
     * hidden 上限 64（tiny 模型足够；需要更大请调大 GRU_MAX_H）。 */
    #define GRU_MAX_H 64
    float rv[GRU_MAX_H], zv[GRU_MAX_H];
    const int H = hidden > GRU_MAX_H ? GRU_MAX_H : hidden;

    for (int k = 0; k < H; ++k) {
        float az = b_ih[H + k] + b_hh[H + k];
        float ar = b_ih[k] + b_hh[k];
        const float *wiz = w_ih + (size_t)(H + k) * input_dim;
        const float *wir = w_ih + (size_t)k * input_dim;
        const float *whz = w_hh + (size_t)(H + k) * H;
        const float *whr = w_hh + (size_t)k * H;
        for (int j = 0; j < input_dim; ++j) {
            ar += wir[j] * x[j];
            az += wiz[j] * x[j];
        }
        for (int j = 0; j < H; ++j) {
            ar += whr[j] * h[j];
            az += whz[j] * h[j];
        }
        rv[k] = 1.0f / (1.0f + expf(-ar));
        zv[k] = 1.0f / (1.0f + expf(-az));
    }

    for (int k = 0; k < H; ++k) {
        const float *win = w_ih + (size_t)(2 * H + k) * input_dim;
        const float *whn = w_hh + (size_t)(2 * H + k) * H;
        float xn = b_ih[2 * H + k];
        float hn = b_hh[2 * H + k];
        for (int j = 0; j < input_dim; ++j) {
            xn += win[j] * x[j];
        }
        for (int j = 0; j < H; ++j) {
            hn += whn[j] * h[j];
        }
        float nv = tanhf(xn + rv[k] * hn);
        h_new[k] = (1.0f - zv[k]) * nv + zv[k] * h[k];
    }
    #undef GRU_MAX_H
}
void layernorm_f32(const float *x, const float *gamma, const float *beta,
                   float *y, int n, float eps) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)x[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < n; ++i) {
        y[i] = ((float)((double)x[i] - mean)) * inv * gamma[i] + beta[i];
    }
}

void attention_f32(const float *q, const float *k, const float *v, float *out,
                   int seq_len, int d_model, float *scores) {
    const float scale = 1.0f / sqrtf((float)d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            const float *qi = q + (size_t)i * d_model;
            const float *kj = k + (size_t)j * d_model;
            for (int d = 0; d < d_model; ++d) {
                dot += qi[d] * kj[d];
            }
            scores[(size_t)i * seq_len + j] = dot * scale;
        }
        /* softmax 第 i 行 */
        float mx = scores[(size_t)i * seq_len];
        for (int j = 1; j < seq_len; ++j) {
            if (scores[(size_t)i * seq_len + j] > mx) {
                mx = scores[(size_t)i * seq_len + j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
            float e = (float)exp((double)(scores[(size_t)i * seq_len + j] - mx));
            scores[(size_t)i * seq_len + j] = e;
            sum += e;
        }
        for (int j = 0; j < seq_len; ++j) {
            scores[(size_t)i * seq_len + j] = (float)(scores[(size_t)i * seq_len + j] / sum);
        }
        for (int d = 0; d < d_model; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                acc += scores[(size_t)i * seq_len + j] * v[(size_t)j * d_model + d];
            }
            out[(size_t)i * d_model + d] = acc;
        }
    }
}

