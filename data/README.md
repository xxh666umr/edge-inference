---
license: mit
task_categories:
  - audio-classification
tags:
  - signal-processing
  - radioml
  - iq-samples
size_categories:
  - n<1K
---

# RadioML 2018 Sample I/Q Dataset

Curated I/Q sample windows for the SignalIQ Modulation Classifier demo.

- **Classes:** 11 modulation types (AM-DSB, AM-SSB, WBFM, QPSK, QAM16, QAM64, 8PSK, BPSK, CPFSK, GFSK, PAM4)
- **Samples:** 550 (50 per class)
- **Fields:** `iq_i`, `iq_q`, `modulation`, `snr_db`, `label`
- **Window length:** 1024 complex samples

Companion dataset for [alirezaaminzadeh/radio-modulation-classifier](https://huggingface.co/alirezaaminzadeh/radio-modulation-classifier).

## RadioML 2016.10a（当前部署版模型使用）

- 来源：Zenodo <https://zenodo.org/records/18397070>（CC BY-NC-SA 4.0，仅学术/非商业用途）
- 下载 `RML2016.10a.tar.bz2`（约 203 MB），解压后得到 `RML2016.10a_dict_optimized.pkl`（约 225 MB）
- 放到本目录（`data/`）即可用 `tools/train_radio2016.py` 训练；该文件已被 `.gitignore` 排除，不入库
- 内容：11 类 × 20 SNR（-20 ~ +18 dB）× 1000 条 = 220,000 条，每条 (2, 128) float32
