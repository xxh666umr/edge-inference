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
