# Cross-Lingual Edge Keyword Spotting (KWS): Phonetic Hard Negative Mining for Bilingual Indic-English Environments

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Export-ONNX%20Opset%2013-lightgrey.svg)](https://onnx.ai/)
[![SRAM Budget](https://img.shields.io/badge/Edge%20Footprint-%3C%20250%20KB%20SRAM-brightgreen.svg)](#comparative-model-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-lightweight ($< 250\text{ KB}$ SRAM footprint), real-time Keyword Spotting (KWS) research framework designed for edge microcontrollers operating in multilingual Indian environments. This repository investigates **Cross-Lingual Phonetic Hard Negative Mining (CL-HNM)**, compact neural architectures (**DS-CNN**, **TC-ResNet8**, and **BC-ResNet**), and temporal streaming post-processing to detect English wake-words (**`"STOP"`**, **`"UP"`**, **`"GO"`**, **`"OFF"`**, **`"ON"`**) while suppressing false activations triggered by phonetically proximate Hindi/Indic speech.

---

## Table of Contents
- [Problem Statement & Motivation](#problem-statement--motivation)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Methodology](#methodology)
  - [1. Cross-Lingual Phonetic Matrix Pre-Check](#1-cross-lingual-phonetic-matrix-pre-check)
  - [2. Multi-Corpus Dataset Curation](#2-multi-corpus-dataset-curation)
  - [3. Acoustic Frontend & Balanced Training](#3-acoustic-frontend--balanced-training)
  - [4. Edge Streaming Post-Processing (Smoothing + NMS)](#4-edge-streaming-post-processing-smoothing--nms)
- [Comparative Model Benchmarks](#comparative-model-benchmarks)
- [Detailed Experimental Results](#detailed-experimental-results)
  - [Experiment A: DS-CNN Continuous Streaming & ONNX Export](#experiment-a-ds-cnn-continuous-streaming--onnx-export)
  - [Experiment B: Custom TC-ResNet8-Style CE vs. CL-HNM Study](#experiment-b-custom-tc-resnet8-style-ce-vs-cl-hnm-study)
  - [Experiment C: BC-ResNet Iso-Parameter Evaluation](#experiment-c-bc-resnet-iso-parameter-evaluation)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Hardware Deployment Notes](#hardware-deployment-notes)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Problem Statement & Motivation

Standard open-source KWS models trained solely on Western English corpora (e.g., Google Speech Commands) suffer severe **False Alarm Rate (FAR)** degradation when deployed in bilingual Indian households or commercial environments. Everyday conversational Hindi contains high-frequency lexical items that share near-identical phoneme sequences with English command keywords (e.g., English *"UP"* `/ʌp/` vs. Hindi *"अप"* `/ap/` and *"उप"* `/up/`; English *"GO"* `/ɡoʊ/` vs. Hindi *"गोल"* `/ɡol/`).

Because always-on microcontroller DSPs (ARM Cortex-M4/M7, ESP32-S3) operate under strict memory ceilings ($\le 250\text{ KB}$ SRAM), simply scaling up model depth is infeasible. This research resolves cross-lingual interference at the **data-centric** and **streaming post-processing** levels while benchmarking sub-250 KB neural architectures independently to select a single optimal edge backbone.

---

## System Architecture & Pipeline

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PHONETIC MATRIX PRE-CHECK (TEXT LEVEL)                │
│  English Keywords (g2p_en -> IPA)  ◄── Normalized Levenshtein (≤ 0.40) ──►        │
│  Indic Transcripts (epitran -> IPA)       37 High-Risk Confusable Seed Words      │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│               PHASE 2: MULTI-CORPUS AUDIO DATA CURATION (16 kHz PCM)              │
│  • Positives (Label 1): Google Speech Commands v2 (+ Svarah Indian-Eng expansion) │
│  • Hard Negatives (Label 0): AI4Bharat Kathbath (+ IndicVoices expansion)         │
│  • Soft Negatives (Label 0): Background Noise & OOV Speech                        │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: ACOUSTIC FRONTEND & BALANCED TRAINING                   │
│  1.0s Audio (16,000 samples) ──► 40-Band Log-Mel Spectrogram [1, 40, 101]         │
│  WeightedRandomSampler + CE Baseline ──► Model-Driven CL-HNM Fine-Tuning          │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│        PHASE 4: COMPARATIVE STUDY (EVALUATED INDEPENDENTLY, < 250 KB RAM)         │
│  Candidate A: DS-CNN (12,546 params | 49.01 KB RAM | 21.29 KB ONNX)               │
│  Candidate B: TC-ResNet8-Style (36,090 params | ~141 KB RAM | 94.48% Macro-F1)    │
│  Candidate C: BC-ResNet (13,178 params | 51.48 KB RAM)                            │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│       PHASE 5: SINGLE WINNER SELECTION, STREAMING NMS & EDGE DEPLOYMENT           │
│  1.0s Sliding Window (20 ms Stride) ──► 5-Frame Moving Avg + 500 ms Debounce NMS  │
│  Result: 95.8% False Alarm Reduction (54.60 FA/hr) ──► ONNX Edge Deployment       │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### 1. Cross-Lingual Phonetic Matrix Pre-Check
Candidate cross-lingual confusions are isolated prior to audio training:
1. English target wake-words are transcribed to International Phonetic Alphabet (IPA) sequences via `g2p_en`.
2. Unique Devanagari vocabulary extracted from **AI4Bharat Kathbath** transcripts ($7,384$ unique tokens) is transliterated to IPA via `epitran`.
3. Pairwise **Normalized Levenshtein Distance** filters out cross-lingual pairs within distance $\le 0.40$, yielding **37 high-risk phonetic seed words**:

| Target English Keyword | Candidate Hindi Collision | Hindi IPA | Normalized Phonetic Distance |
| :--- | :--- | :--- | :---: |
| **GO** | गोल | `/ɡol/` | `0.333` |
| **STOP** | स्लिप | `/slip/` | `0.500` |
| **STOP** | स्तर | `/stər/` | `0.500` |
| **UP** | अप | `/ap/` | `0.500` |
| **UP** | उप | `/up/` | `0.500` |
| **ON** | इन | `/in/` | `0.500` |
| **ON** | उन | `/un/` | `0.500` |
| **STOP** | सोच | `/sot͡ʃ/` | `0.600` |

### 2. Multi-Corpus Dataset Curation
All audio clips are standardized to $16\text{ kHz}$ mono PCM and segmented into $1.0\text{ s}$ ($16,000$-sample) windows:

| Dataset | Role in Pipeline | Status & Coverage |
| :--- | :--- | :--- |
| **[Google Speech Commands v2](https://arxiv.org/abs/1804.03209)** | English Keyword Positives (`Label 1`) & Noise (`Label 0`) | **Completed:** $19,065$ 5-keyword clips (DS-CNN/BC-ResNet) and $85,380$ train / $11,005$ test clips across all 35 GSC classes (TC-ResNet8 binary study) + background noise. |
| **[AI4Bharat Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath)** | Scripted Hindi Hard Negatives (`Label 0`) | **Completed:** $433$ verified 1.0s phonetic slices (DS-CNN) and $698$ speaker-disjoint clips from 49 speakers (TC-ResNet8). |
| **[AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)** | Indian-English Positives (`1`) & OOV Negatives (`0`) | **Parallel Expansion (`03_mine_parallel.py`):** Read & spontaneous English across 117 Indian speakers (19 states). |
| **[AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)** | Conversational Hindi Hard Negatives (`Label 0`) | **Parallel Expansion (`03_mine_parallel.py`):** $>93\%$ extempore natural Hindi speech for conversational robustness. |

### 3. Acoustic Frontend & Balanced Training
- **Log-Mel Spectrogram Extraction:** $16\text{ kHz}$ waveforms ($16,000$ samples) are transformed using a $25\text{ ms}$ Hann window (`n_fft=400`), $10\text{ ms}$ stride (`hop_length=160`), and $40$ Mel filterbanks, producing an input tensor of shape `[1, 40, 101]`.
- **Dynamic Class Balancing & CL-HNM:** A PyTorch `WeightedRandomSampler` prevents majority-class collapse during training—enforcing a $50/50$ positive/negative split in the DS-CNN pipeline and a `50% English / 35% Mined Hard Hindi / 10% Other Hindi / 5% Noise` split during TC-ResNet8 CL-HNM fine-tuning.

### 4. Edge Streaming Post-Processing (Smoothing + NMS)
Evaluating a continuous $16\text{ kHz}$ audio stream at a $20\text{ ms}$ hop ($50\text{ Hz}$ MCU buffer update rate) causes a single $400\text{ ms}$ confusable Hindi word to overlap $\sim 20$ consecutive sliding windows. Without post-processing, 2 spoken words inflate into $48$ frame-level false alarms ($1,310.42\text{ FA/hr}$).
1. **Posterior Moving Average:** Smooths class-1 softmax probabilities over a rolling **5-frame ($100\text{ ms}$) buffer** to reject transient spectral spikes.
2. **Non-Maximum Suppression (Debouncing):** Enforces a **25-frame ($500\text{ ms}$) refractory lockout** immediately following any trigger event above threshold $\tau = 0.85$.

---

## Comparative Model Benchmarks

All three candidate architectures are evaluated independently under the **$< 250\text{ KB}$ microcontroller SRAM constraint**. Note that each architecture was benchmarked on its corresponding evaluation protocol (continuous streaming for DS-CNN; 11,085-clip binary held-out test set for TC-ResNet8; 418-clip 6-class split for BC-ResNet):

| Architecture | Design Paradigm | Trainable Params | FP32 Parameter RAM | Export / Checkpoint Size | Key Evaluation Metric |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DS-CNN** *(Baseline)* | 2D Depthwise Separable Conv | `12,546` | `49.01 KB` | **`21.29 KB`** *(FP32 ONNX)* | **`54.60 FA/hr`** debounced streaming FAR (`99.93%` train acc.) |
| **TC-ResNet8-Style** *(CL-HNM)* | 1D Temporal Conv + Residual Skips | `36,090` | `~141.00 KB` | `~482 KB` *(`.pth` checkpoint)* | **`94.48%` Macro-F1** / **`86.25%` Neg. Recall** (`99.85%` test acc. on 11,085 clips) |
| **BC-ResNet** *(Iso-Parameter)* | 2D Freq-DW + 1D Broadcasted Residual | `13,178` | `51.48 KB` | `~53 KB` *(est. ONNX)* | **`100.0%` Test Acc.** / `0.17` Macro-F1 *(418 `UNKNOWN`-only test clips; see note below)* |

> **Quantization Finding:** For ultra-compact networks under $50\text{ KB}$ ($\approx 12.5\text{k}$ parameters), dynamic INT8 ONNX quantization increases binary size from `21.29 KB` to `30.21 KB` due to injected `QuantizeLinear`/`DequantizeLinear` graph nodes exceeding weight compression savings. Exporting native **FP32 ONNX (`21.29 KB`)** or using **TFLite Static INT8** is optimal for deployment.

---

## Detailed Experimental Results

### Experiment A: DS-CNN Continuous Streaming & ONNX Export
Evaluated on continuous $16\text{ kHz}$ Kathbath Hindi audio streams ($30$ streams, $131.87\text{ seconds}$ total duration) at a $50\text{ Hz}$ sliding window rate:
- **Unfiltered Frame Triggers:** $48$ raw triggers ($1,310.42\text{ FA/hr}$).
- **Debounced & Smoothed Triggers:** Reduced to **$2$ spoken word events** (**$54.60\text{ FA/hr}$**, a **$95.8\%$ reduction**).
- **Production Binary:** Exported to `exported_models/edge_kws_fp32.onnx` at **`21.29 KB`** ($< 9\%$ of the $250\text{ KB}$ SRAM limit).

### Experiment B: Custom TC-ResNet8-Style CE vs. CL-HNM Study
A 36,090-parameter temporal convolutional residual network was trained for 10 epochs with Cross-Entropy (`CE`, best checkpoint at Epoch 8 with `81.85%` val Macro-F1), used to mine the top-100 most confusing Hindi training clips, and fine-tuned for 3 epochs with **Cross-Lingual Hard Negative Mining (`CL-HNM`)**.

Both models were evaluated on the same **11,085 held-out test clips** ($11,005$ English keywords, $79$ speaker-disjoint Hindi clips, $1$ noise clip) at validation-selected decision threshold **`0.03`**:

| Metric | CE Baseline (Epoch 8) | CL-HNM Fine-Tuned (Epoch 3) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | `99.83%` | **`99.85%`** | `+0.02%` |
| **English Precision** | `99.86%` | **`99.90%`** | `+0.04%` |
| **English Recall** | **`99.96%`** | `99.95%` | `-0.01%` |
| **Negative Recall (Hindi Rejection)** | `81.25%` | **`86.25%`** | **`+5.00%`** |
| **False Acceptances (Hindi $\rightarrow$ Keyword)** | `15` | **`11`** | **`-4 errors`** |
| **False Rejections (Missed Keyword)** | **`4`** | `6` | `+2 errors` |
| **Macro-F1 Score** | `93.58%` | **`94.48%`** | **`+0.90%`** |

**Held-Out Confusion Matrices (`Threshold = 0.03`):**
- **CE Baseline:** `TN = 65`, `FP = 15`, `FN = 4`, `TP = 11,001`
- **CL-HNM Fine-Tuned:** `TN = 69`, `FP = 11`, `FN = 6`, `TP = 10,999`

### Experiment C: BC-ResNet Iso-Parameter Evaluation
- **Architecture Footprint:** `13,178` parameters (`51.48 KB` FP32 RAM), matching DS-CNN's memory footprint while combining 2D frequency-depthwise convolutions with 1D temporal broadcasted residuals.
- **Recorded Run Diagnostics:** Trained for 30 epochs (`~99.52%` validation accuracy; validation loss between `0.035` and `0.083`). On the 6-class evaluation split (`STOP`, `UP`, `GO`, `OFF`, `ON`, `UNKNOWN`), all `418` test samples belonged to the `UNKNOWN` class (`100.0%` overall accuracy, `1.00` `UNKNOWN` F1, `0.17` macro-average F1). Re-evaluating BC-ResNet on a balanced multi-keyword test split is required for direct per-keyword F1 comparison.

---

## Repository Structure

```text
cross-lingual-edge-kws/
├── checkpoints/
│   ├── edge_kws_hnm.pth                       # Trained DS-CNN FP32 weights (12,546 params)
│   ├── tc_resnet8.pth                         # Trained TC-ResNet8-style weights (36,090 params)
│   └── bc_resnet.pth                          # Trained BC-ResNet weights (13,178 params)
├── exported_models/
│   └── edge_kws_fp32.onnx                     # Production 21.29 KB ONNX graph (Opset 13)
├── notebooks/
│   ├── 01_train_dscnn.ipynb                   # DS-CNN training, streaming NMS & ONNX export
│   ├── 02_train_tcresnet8.ipynb               # TC-ResNet8 CE vs. CL-HNM experiment & plots
│   └── 03_train_bcresnet.ipynb                # BC-ResNet 6-class training & confusion matrix
├── Research_Graphs/
│   ├── Figure_1_Test_Performance_Comparison.png
│   ├── Figure_2_Confusion_Matrices.png
│   ├── Figure_3_Threshold_Performance.png
│   ├── Figure_4A_Training_Validation_Loss.png
│   ├── Figure_4B_Validation_Macro_F1.png
│   └── Final_Research_Results.csv
├── scripts/
│   ├── 01_phonetic_mining.py                  # IPA conversion & Levenshtein collision matrix
│   ├── 02_slice_and_verify.py                 # GSC download + 1.0s Kathbath audio slicing
│   ├── 03_mine_parallel.py                    # Streaming IndicVoices & Svarah parallel ingestion
│   ├── models.py                              # LightweightEdgeKWS (DS-CNN), TCResNet8, BCResNet
│   ├── train.py                               # Balanced training loop (WeightedRandomSampler)
│   ├── eval_stream.py                         # 50 Hz continuous streaming evaluator with NMS
│   └── export_onnx.py                         # ONNX graph export & footprint verification
├── hard_negative_seed_words.csv               # 37 mined Hindi-English phonetic collision pairs
├── requirements.txt
└── README.md
```

---

## Installation & Quickstart

### 1. Environment Setup
```bash
git clone [https://github.com/](https://github.com/)<your-username>/cross-lingual-edge-kws.git
cd cross-lingual-edge-kws

pip install torch torchaudio datasets pandas numpy epitran g2p-en python-Levenshtein onnx onnxruntime huggingface_hub
```

### 2. Authenticate Hugging Face (For Gated AI4Bharat Streams)
Accept the CC-BY-4.0 dataset terms on [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) and [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), then authenticate via CLI:
```bash
huggingface-cli login
```

### 3. Run the End-to-End Pipeline
```bash
# Step 1: Generate phonetic collision dictionary (hard_negative_seed_words.csv)
python scripts/01_phonetic_mining.py

# Step 2: Download Google Speech Commands v2 & extract Kathbath 1.0s hard negatives
python scripts/02_slice_and_verify.py

# Step 3: Stream IndicVoices (Hindi) & Svarah (Indian English) in parallel
python scripts/03_mine_parallel.py

# Step 4: Train & benchmark candidate models
python scripts/train.py

# Step 5: Evaluate continuous streaming False Alarm Rate (FA/hr) with NMS debouncing
python scripts/eval_stream.py

# Step 6: Export production ONNX binary (< 250 KB)
python scripts/export_onnx.py
```

---

## Hardware Deployment Notes

- **Microcontrollers with Hardware FPU (STM32F4/F7/H7, ESP32-S3, RP2350):** Deploy `exported_models/edge_kws_fp32.onnx` (`21.29 KB`) directly via **STM32 X-CUBE-AI**, **ESP-DL**, or **ONNX Runtime Mobile** with zero quantization degradation.
- **Integer-Only Microcontrollers (ARM Cortex-M0/M3):** Convert the selected PyTorch checkpoint to **TFLite Static INT8** using a 100-sample representative calibration set to fuse scale/zero-point constants without dynamic node overhead (`~14 KB` flash footprint).

---

## Limitations & Future Work

1. **Unified Apple-to-Apple Test Suite:** Re-evaluate `BC-ResNet` on a balanced multi-keyword test split (fixing label-case mapping so target keywords are populated alongside `UNKNOWN`) and benchmark all three checkpoints on the exact same continuous streaming test set.
2. **Extended Continuous Streaming Hours:** Scale streaming FAR evaluation from $131.87\text{ s}$ ($30$ files) to $500+$ continuous streams ($\ge 35\text{ minutes}$) across **Kathbath + IndicVoices + Svarah** to tighten statistical confidence intervals on `FA/hr`.
3. **On-Device Profiling:** Measure peak SRAM usage, inference latency ($\text{ms}$), and energy per inference directly on target STM32 / ESP32-S3 hardware.

---

## References

1. **DS-CNN:** Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). *Hello Edge: Keyword Spotting on Microcontrollers*. arXiv:1711.07128.
2. **TC-ResNet:** Choi, S., Seo, S., Shin, B., Byun, H., Kersner, M., Kim, B., Kim, D., & Ha, S. (2019). *Temporal Convolution for Real-time Keyword Spotting on Mobile Devices*. INTERSPEECH 2019.
3. **BC-ResNet:** Kim, B., Chang, S., Lee, J., & Sung, D. (2021). *Broadcasted Residual Learning for Efficient Keyword Spotting*. INTERSPEECH 2021.
4. **Google Speech Commands:** Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*. arXiv:1804.03209.
5. **Kathbath & IndicVoices:** Javed, T., et al. (2023/2024). *Towards Building ASR Systems for the Next Billion Users* & *IndicVoices: Towards Building an Inclusive Multilingual Speech Dataset for Indian Languages*. AI4Bharat.
6. **Svarah:** Javed, T., et al. (2023). *Svarah: Evaluating English ASR Systems on Indian Accents*. AI4Bharat.
