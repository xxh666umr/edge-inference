/* STM32F103ZET6 裸机固件：int8 CNN 推理 + DWT CYCCNT 时延统计
 *
 * 串口：USART1 (PA9=TX, PA10=RX)，115200 8N1，printf 重定向到 _write()。
 *       BRR 由 SystemCoreClock / APB2 预分频运行时计算，因此启动文件里的
 *       SystemInit() 配成 8MHz HSI 或 72MHz PLL 都不用改本文件；实际时钟
 *       频率从 SystemCoreClock（system_stm32f1xx.c 运行时更新）读取。
 * 推理：models/model.c 的 model_forward()（接口冻结，见 models/model.h）：
 *       int8 输入 [MODEL_INPUT_C][MODEL_INPUT_LEN] 行主序（I 通道在前），
 *       返回 argmax 类别并填 MODEL_NUM_CLASSES 个 float 概率。
 * 计时：Cortex-M3 DWT->CYCCNT；warmup 1 次后连续 TIMING_RUNS 次统计
 *       min/avg/max cycles，并按 SystemCoreClock 换算成 us。
 * 资源：所有 buffer 静态分配（.bss），无动态内存、无中断依赖。
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "stm32f1xx.h"     /* CMSIS：RCC / GPIO / USART1 / CoreDebug / DWT */
#include "nn.h"            /* int8 算子（分阶段 benchmark 用） */
#include "model.h"         /* 冻结接口：MODEL_* 宏 + model_forward() */
#include "demo_sample.h"   /* 固件内固定演示样本：DEMO_INPUT / DEMO_TRUE_CLASS / DEMO_PRED_CLASS */

/* 链接脚本中 .ramfunc 段的 RAM 运行地址与 Flash 加载地址（启动时拷贝） */
extern uint8_t _siramfunc[];
extern uint8_t _sramfunc[];
extern uint8_t _eramfunc[];

#define USART_BAUD        115200u
#define TIMING_RUNS       20u    /* 单帧时延统计次数（20 次约 0.7 s，便于 1 Hz 实时看板） */
#define REPORT_PERIOD_S   1u    /* 报告周期（秒） */

/* ============================ 系统时钟 ============================ */

/* 尝试把 SYSCLK 配到 72MHz（8MHz HSE + PLL×9）；无 HSE 时保持 HSI 8MHz。
 * 不修改 cmsis/system_stm32f1xx.c（其 SystemInit 只做最小初始化），
 * 全部在 main 里显式配置，失败可回退，不会卡死。
 * 返回 1 = 72MHz PLL，0 = 回退 8MHz HSI。 */
static int clock_init_72mhz(void) {
    uint32_t timeout;

    RCC->CR |= RCC_CR_HSEON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_HSERDY) == 0u) && (--timeout != 0u)) {
    }
    if ((RCC->CR & RCC_CR_HSERDY) == 0u) {
        SystemCoreClockUpdate();   /* 仍为 HSI 8MHz */
        return 0;
    }

    /* 72MHz 需要 2 个 Flash 等待周期（先设，再升频） */
    FLASH->ACR = FLASH_ACR_PRFTBE | FLASH_ACR_LATENCY_2;

    /* AHB=/1，APB1=/2（36MHz 上限），APB2=/1（72MHz） */
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
              | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV2 | RCC_CFGR_PPRE2_DIV1;
    /* PLL 源 = HSE（不分频），倍频 ×9 */
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_PLLSRC | RCC_CFGR_PLLXTPRE | RCC_CFGR_PLLMULL))
              | RCC_CFGR_PLLSRC | RCC_CFGR_PLLMULL9;

    RCC->CR |= RCC_CR_PLLON;
    timeout = 0x50000u;
    while (((RCC->CR & RCC_CR_PLLRDY) == 0u) && (--timeout != 0u)) {
    }
    if ((RCC->CR & RCC_CR_PLLRDY) == 0u) {
        SystemCoreClockUpdate();
        return 0;
    }

    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    timeout = 0x50000u;
    while (((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) && (--timeout != 0u)) {
    }
    SystemCoreClockUpdate();       /* 从寄存器重算，保证 BRR/µs 换算正确 */
    return (RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL;
}

/* ============================== USART1 ============================== */

/* USART1 挂在 APB2 上，PCLK2 = SystemCoreClock / APB2 预分频。
 * RCC_CFGR.PPRE2[2:0]：0xx = /1，100 = /2，101 = /4，110 = /8，111 = /16。
 * 这样 8MHz HSI（默认 SystemInit 配置）与 72MHz PLL 都能得到正确波特率。 */
static uint32_t usart1_pclk2(void) {
    const uint32_t ppre2 = (RCC->CFGR & RCC_CFGR_PPRE2) >> RCC_CFGR_PPRE2_Pos;
    if (ppre2 < 4u) {
        return SystemCoreClock;              /* /1 */
    }
    return SystemCoreClock >> (ppre2 - 3u);  /* /2, /4, /8, /16 */
}

static void usart1_init(void) {
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN | RCC_APB2ENR_USART1EN;

    /* PA9：复用推挽输出 50MHz，CRH[7:4] = MODE(11) CNF(10) = 0b1011 */
    GPIOA->CRH = (GPIOA->CRH & ~(0xFu << 4)) | (0xBu << 4);
    /* PA10：浮空输入，CRH[11:8] = MODE(00) CNF(01) = 0b0100 */
    GPIOA->CRH = (GPIOA->CRH & ~(0xFu << 8)) | (0x4u << 8);

    /* 8N1：CR2/CR3 保持复位值，1 停止位 / 8 数据位 / 无校验 */
    USART1->CR2 = 0u;
    USART1->CR3 = 0u;
    /* BRR = round(PCLK2 / baud)（OVER8=0，16 倍过采样）：72MHz -> 625，8MHz -> 69 */
    {
        const uint32_t pclk2 = usart1_pclk2();
        uint32_t brr = (pclk2 + USART_BAUD / 2u) / USART_BAUD;
        if (brr == 0u) {
            brr = 1u;
        }
        USART1->BRR = brr;
    }
    USART1->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
}

/* printf 重定向：newlib 的 printf 最终调用 _write()。轮询 TXE，阻塞式发送。 */
int _write(int file, char *ptr, int len) {
    (void)file;
    for (int i = 0; i < len; ++i) {
        while ((USART1->SR & USART_SR_TXE) == 0u) {
        }
        USART1->DR = (uint8_t)ptr[i];
    }
    return len;
}

/* ========================= DWT 周期计数器 ========================= */

static int dwt_ok = 0;   /* DWT CYCCNT 是否成功使能 */

static void dwt_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  /* 打开 trace 单元 */
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;             /* 使能周期计数 */
    dwt_ok = (DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0u;
}

static inline uint32_t dwt_cycles(void) { return DWT->CYCCNT; }

static void delay_cycles(uint32_t cycles) {
    const uint32_t start = dwt_cycles();
    while ((uint32_t)(dwt_cycles() - start) < cycles) {
    }
}

/* ============================ 推理 + 计时 ============================ */

static int8_t input_buf[MODEL_INPUT_C * MODEL_INPUT_LEN];
static float probs[MODEL_NUM_CLASSES];

/* 遥测寄存器区：无串口时可用 OpenOCD 直接读 RAM 获取时延结果（地址见 .map / nm）：
 *   [0] magic 'EDGE'  [1] SystemCoreClock  [2] 时钟源(0=HSI,1=HSE,2=PLL)
 *   [3] argmax         [4] avg cycles       [5] min cycles
 *   [6] max cycles     [7] timing runs
 * 例如：openocd ... -c "init; halt; mdw 0x200000xx 8; resume; exit" */
volatile uint32_t g_telemetry[8] = {0};

/* ===================== 分阶段耗时剖析（上板实测） =====================
 * 用与真实模型完全相同形状的 dummy 权重/缓冲区，逐算子计时（100 次取平均），
 * 结果放 g_bench，可由串口打印或 OpenOCD 读 RAM。用来定位推理总时延的构成。
 * 权重全 0 只影响数值、不影响循环次数，因此计时有效。 */
#define BENCH_REPEAT 100
volatile uint32_t g_bench[12] = {0};   /* 0 conv1 1 req1 2 pool1 3 conv2 4 req2
                                          5 pool2 6 avg 7 fc 8 softmax 9 memset 校准
                                          10 requantize 浮点版  11 requantize 定点版 */

static int32_t b_acc[960];      /* conv1 acc: 16*60 = 960（各阶段复用） */
static int8_t  b_i8[960];
static int8_t  b_pool1[480];    /* conv1 输出池化后: 16*30 */
static int8_t  b_pool2[192];    /* conv2 输出池化后: 32*6 */
static int8_t  b_gap[32];
static int32_t b_logits[11];
static float   b_probs[11];
static int8_t  b_input[2 * 126];

static const int8_t  BENCH_W1[224]  = {0};
static const int32_t BENCH_B1[16]   = {0};
static const int8_t  BENCH_W2[2560] = {0};
static const int32_t BENCH_B2[32]   = {0};
static const int8_t  BENCH_WF[352]  = {0};
static const int32_t BENCH_BF[11]   = {0};

#define TIME_IT(idx, expr)                                                    \
    do {                                                                      \
        uint32_t _t0 = dwt_cycles();                                          \
        for (int _r = 0; _r < BENCH_REPEAT; ++_r) {                           \
            (void)(expr);                                                     \
        }                                                                     \
        g_bench[idx] = (dwt_cycles() - _t0) / BENCH_REPEAT;                   \
    } while (0)

static void bench_stages(void) {
    memcpy(b_input, DEMO_INPUT, sizeof(b_input));
    TIME_IT(9, memset(b_i8, 0, sizeof(b_i8)));
    TIME_IT(0, conv1d_accumulate(BENCH_W1, BENCH_B1, b_input, b_acc, 2, 16, 126, 7, 2));
    TIME_IT(10, requantize_relu(b_acc, b_i8, 960, 0.00245f));                  /* 旧：软浮点 */
    TIME_IT(11, requantize_relu_fx(b_acc, b_i8, 960, MODEL_CONV1_FX_MUL,       /* 新：定点 */
                                   MODEL_CONV1_FX_SHIFT));
    TIME_IT(1, requantize_relu_fx(b_acc, b_i8, 960, MODEL_CONV1_FX_MUL,
                                  MODEL_CONV1_FX_SHIFT));
    TIME_IT(2, maxpool1d(b_i8, b_pool1, 16, 60, 2, 2));
    TIME_IT(3, conv1d_accumulate(BENCH_W2, BENCH_B2, b_pool1, b_acc, 16, 32, 30, 5, 2));
    TIME_IT(4, requantize_relu_fx(b_acc, b_i8, 416, MODEL_CONV2_FX_MUL,
                                  MODEL_CONV2_FX_SHIFT));
    TIME_IT(5, maxpool1d(b_i8, b_pool2, 32, 13, 2, 2));
    TIME_IT(6, avgpool1d(b_pool2, b_gap, 32, 6, 6, 1));
    TIME_IT(7, matmul_accumulate(BENCH_WF, BENCH_BF, b_gap, b_logits, 32, 11));
    TIME_IT(8, softmax_argmax(b_logits, 11, 0.000442f, b_probs));
}

typedef struct {
    uint32_t min_cycles;
    uint32_t max_cycles;
    uint64_t sum_cycles;
    uint32_t runs;
} latency_stats_t;

/* 先 warmup 1 次（排除首次调用开销），再连续 TIMING_RUNS 次计时。
 * model_forward() 每次输入相同，最后一次的 probs 即最终分类概率。 */
static latency_stats_t measure_latency(void) {
    latency_stats_t st = {0xFFFFFFFFu, 0u, 0u, TIMING_RUNS};

    (void)model_forward(input_buf, probs);   /* warmup，不计入统计 */

    for (uint32_t i = 0; i < TIMING_RUNS; ++i) {
        const uint32_t t0 = dwt_cycles();
        (void)model_forward(input_buf, probs);
        const uint32_t dt = dwt_cycles() - t0;
        if (dt < st.min_cycles) {
            st.min_cycles = dt;
        }
        if (dt > st.max_cycles) {
            st.max_cycles = dt;
        }
        st.sum_cycles += dt;
    }
    return st;
}

static const char *clock_source_name(void) {
    switch ((RCC->CFGR & RCC_CFGR_SWS) >> RCC_CFGR_SWS_Pos) {
        case 0u: return "HSI";
        case 1u: return "HSE";
        case 2u: return "PLL";
        default: return "unknown";
    }
}

static void print_report(void) {
    const latency_stats_t st = measure_latency();
    const unsigned sysclk = (unsigned)SystemCoreClock;
    const unsigned avg_cycles = (unsigned)((st.sum_cycles + st.runs / 2u) / st.runs);
    const int pred = (int)model_forward(input_buf, probs);   /* 取与本次报告一致的结果 */

    printf("\r\n========================================\r\n");
    printf("model  : %s\r\n", MODEL_NAME);
    printf("input  : C=%d L=%d int8, classes=%d, params=%d\r\n",
           (int)MODEL_INPUT_C, (int)MODEL_INPUT_LEN,
           (int)MODEL_NUM_CLASSES, (int)MODEL_PARAM_COUNT);
    printf("clock  : %u Hz (%s), dwt=%s\r\n",
           sysclk, clock_source_name(), dwt_ok ? "ok" : "FAIL");
    printf("demo   : true=%d, pc_ref_pred=%d\r\n",
           (int)DEMO_TRUE_CLASS, (int)DEMO_PRED_CLASS);

    for (int i = 0; i < (int)MODEL_NUM_CLASSES; ++i) {
        printf("  class %2d: prob=%.4f\r\n", i, probs[i]);
    }
    printf("pred   : %d (ref %d) %s\r\n", pred, (int)DEMO_PRED_CLASS,
           pred == (int)DEMO_PRED_CLASS ? "MATCH" : "MISMATCH");
    printf("latency: min=%u avg=%u max=%u cycles (N=%u)\r\n",
           (unsigned)st.min_cycles, avg_cycles, (unsigned)st.max_cycles,
           (unsigned)st.runs);
    printf("latency: min=%.2f avg=%.2f max=%.2f us\r\n",
           (float)st.min_cycles * 1.0e6f / (float)sysclk,
           (float)avg_cycles * 1.0e6f / (float)sysclk,
           (float)st.max_cycles * 1.0e6f / (float)sysclk);
    printf("========================================\r\n");

    /* 发布到遥测区，便于 OpenOCD 无串口读取（见文件头注释） */
    {
        const uint32_t sws = (RCC->CFGR & RCC_CFGR_SWS) >> RCC_CFGR_SWS_Pos;
        g_telemetry[0] = 0x45444745u;   /* 'EDGE' */
        g_telemetry[1] = (uint32_t)sysclk;
        g_telemetry[2] = sws;
        g_telemetry[3] = (uint32_t)pred;
        g_telemetry[4] = (uint32_t)avg_cycles;
        g_telemetry[5] = (uint32_t)st.min_cycles;
        g_telemetry[6] = (uint32_t)st.max_cycles;
        g_telemetry[7] = (uint32_t)st.runs;
    }
}

int main(void) {
    const int pll72 = clock_init_72mhz();   /* 尝试 72MHz，失败则 8MHz HSI */
    usart1_init();
    dwt_init();

    /* 把 .ramfunc（conv/pool/fc/重量化热循环）从 Flash 拷到 SRAM 执行。
     * 链接脚本中 .ramfunc 的 VMA 在 RAM、LMA 在 Flash，必须在使用前拷贝。 */
    memcpy(_sramfunc, _siramfunc, (size_t)(_eramfunc - _sramfunc));

    memcpy(input_buf, DEMO_INPUT, sizeof(input_buf));   /* Flash -> RAM 演示输入 */

    printf("\r\n[edge-inference] STM32F103ZET6 int8 CNN firmware\r\n");
    printf("build  : %s %s\r\n", __DATE__, __TIME__);
    printf("clock  : %u Hz (%s), pll72=%d\r\n", (unsigned)SystemCoreClock,
           clock_source_name(), pll72);
    printf("model  : %s, input %d x %d int8, classes %d, params %d\r\n",
           MODEL_NAME, (int)MODEL_INPUT_C, (int)MODEL_INPUT_LEN,
           (int)MODEL_NUM_CLASSES, (int)MODEL_PARAM_COUNT);

    /* 一次性的分阶段耗时剖析（与真实模型同形状，见 bench_stages 注释） */
    bench_stages();
    printf("--- stage cycles, avg of %d runs ---\r\n", BENCH_REPEAT);
    printf("bench1 : conv1=%u req1=%u pool1=%u conv2=%u req2=%u\r\n",
           (unsigned)g_bench[0], (unsigned)g_bench[1], (unsigned)g_bench[2],
           (unsigned)g_bench[3], (unsigned)g_bench[4]);
    printf("bench2 : pool2=%u avg=%u fc=%u softmax=%u memset960B=%u\r\n",
           (unsigned)g_bench[5], (unsigned)g_bench[6], (unsigned)g_bench[7],
           (unsigned)g_bench[8], (unsigned)g_bench[9]);
    printf("bench3 : requant_float=%u requant_fx=%u (960 元素)\r\n",
           (unsigned)g_bench[10], (unsigned)g_bench[11]);

    for (;;) {
        print_report();
        delay_cycles(REPORT_PERIOD_S * SystemCoreClock);
    }
}


