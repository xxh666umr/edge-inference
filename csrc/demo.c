/* PC 端最小示例：用 models/model.c 里导出的 int8 CNN 对一条真实调制识别样本推理。
 *
 * 与 tools/train_modrec.py 生成的 model.h / model.c / demo_sample.h 配套使用。
 * 编译（在仓库根目录，MSVC x64）：
 *   cmd /c "call \"...\vcvars64.bat\" && cl /nologo /O2 /utf-8 /Icsrc /Imodels \
 *          csrc\nn.c models\model.c csrc\demo.c /Fe:demo.exe"
 * 运行：demo.exe
 */
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "nn.h"
#include "model.h"
#include "demo_sample.h"

int main(void) {
    static float probs[MODEL_NUM_CLASSES];
    const int repeats = 1000;

    /* 预热，同时打印单次分类结果 */
    int arg = model_forward(DEMO_INPUT, probs);

    printf("model     : %s\n", MODEL_NAME);
    printf("input     : %d x %d int8 (params=%d)\n",
           MODEL_INPUT_C, MODEL_INPUT_LEN, MODEL_PARAM_COUNT);
    printf("predicted : %d (%s)   true: %d (%s)\n",
           arg, DEMO_PRED_NAME, DEMO_TRUE_CLASS, DEMO_TRUE_NAME);
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        printf("  class %2d : %.4f\n", i, (double)probs[i]);
    }

    /* PC 端粗略计时（仅作趋势参考，MCU 实测见固件 DWT 输出） */
    clock_t t0 = clock();
    for (int r = 0; r < repeats; ++r) {
        arg = model_forward(DEMO_INPUT, probs);
    }
    clock_t t1 = clock();
    double us = 1e6 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC / repeats;
    printf("avg forward time on PC: %.1f us (%d runs, argmax=%d)\n", us, repeats, arg);
    return 0;
}
