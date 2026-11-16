# Cross-Lingual Edge Keyword Spotting (KWS): Phonetic Hard Negative Mining for Bilingual Indic-English Environments

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/Export-ONNX%20Opset%2013-lightgrey.svg)](https://onnx.ai/)
[![SRAM Budget](https://img.shields.io/badge/Edge%20Footprint-%3C%20250%20KB%20SRAM-brightgreen.svg)](#comparative-model-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-lightweight ($< 250\text{ KB}$ SRAM footprint), real-time Keyword Spotting (KWS) framework designed for edge microcontrollers operating in multilingual Indian environments. This repository introduces a **Cross-Lingual Phonetic Hard Negative Mining** pipeline and edge post-processing engine that enables compact neural networks to detect English wake-words (**`"STOP"`**, **`"UP"`**, **`"GO"`**, **`"OFF"`**, **`"ON"`**) while suppressing false alarms triggered by phonetically proximate native Hindi/Indic speech.

---

## Table of Contents
- [Problem Statement & Motivation](#problem-statement--motivation)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Methodology](#methodology)
  - [1. Cross-Lingual Phonetic Matrix Pre-Check](#1-cross-lingual-phonetic-matrix-pre-check)
  - [2. Multi-Corpus Parallel Dataset Curation](#2-multi-corpus-parallel-dataset-curation)
  - [3. Acoustic Frontend & Balanced Training](#3-acoustic-frontend--balanced-training)
  - [4. Edge Streaming Post-Processing (Smoothing + NMS)](#4-edge-streaming-post-processing-smoothing--nms)
- [Comparative Model Benchmarks](#comparative-model-benchmarks)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Hardware Deployment Notes](#hardware-deployment-notes)
- [References](#references)

---

## Problem Statement & Motivation

Standard open-source KWS models trained solely on Western English corpora (e.g., Google Speech Commands) suffer severe **False Alarm Rate (FAR)** degradation when deployed in bilingual Indian households or commercial environments. Everyday conversational Hindi contains high-frequency lexical items that share near-identical phoneme sequences with English command keywords (e.g., English *"UP"* `/ʌp/` vs. Hindi *"अप"* `/ap/` and *"उप"* `/up/`; English *"GO"* `/ɡoʊ/` vs. Hindi *"गोल"* `/ɡol/`).

Because always-on microcontroller DSPs (ARM Cortex-M4/M7, ESP32-S3) operate under strict memory ceilings ($\le 250\text{ KB}$ SRAM), simply scaling up model depth is infeasible. This research solves cross-lingual interference at the **data-centric** and **streaming post-processing** levels while benchmarking sub-250 KB neural architectures.

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
│               PHASE 2: MULTI-CORPUS PARALLEL AUDIO MINING (16 kHz PCM)            │
│  • Positives (Label 1): Google Speech Commands v2 + AI4Bharat Svarah (Indian Eng) │
│  • Hard Negatives (Label 0): AI4Bharat Kathbath + IndicVoices (1s Sliced Clips)   │
│  • Soft Negatives (Label 0): Svarah English OOV + GSC Background Noise            │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 3: ACOUSTIC FRONTEND & BALANCED TRAINING                   │
│  1.0s Audio (16,000 samples) ──► 40-Band Log-Mel Spectrogram [1, 40, 101]         │
│  WeightedRandomSampler (50/50 Batch Split) + AdamW + Cosine Annealing             │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│             PHASE 4: COMPARATIVE SUB-250 KB ARCHITECTURES & ONNX EXPORT           │
│  1. DS-CNN (Baseline: 49.01 KB RAM | 21.29 KB ONNX)                               │
│  2. TC-ResNet8 (1D Temporal Residual: 130.66 KB RAM)                              │
│  3. BC-ResNet (Iso-Parameter 2D/1D Broadcasted Residual: 51.48 KB RAM)            │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│             PHASE 5: 50 Hz CONTINUOUS STREAMING & EDGE POST-PROCESSING            │
│  1.0s Sliding Window (20 ms Stride) ──► 5-Frame Moving Average (100 ms)           │
│                                     ──► 500 ms Debounce Lockout (NMS)             │
│                                     ──► 95.8% False Alarm Reduction               │
└───────────────────────────────────────────────────────────────────────────────────┘
```
## Methodology

### 1. Cross-Lingual Phonetic Matrix Pre-Check

Rather than blindly ingesting thousands of hours of raw speech, candidate traps are isolated prior to audio streaming:

1. English target wake-words are transcribed to International Phonetic Alphabet (IPA) sequences via `g2p_en`.
2. Unique Devanagari vocabulary extracted from **AI4Bharat Kathbath** transcripts ($7,384$ unique tokens) is transliterated to IPA via `epitran`.
3. Pairwise **Normalized Levenshtein Distance** filters out cross-lingual pairs within distance $\le 0.40$, yielding **37 high-risk phonetic seed words**:

| Target English Keyword | Mined Hindi Collision | Hindi IPA | Normalized Phonetic Distance |
| --- | --- | --- | --- |
| **GO** | गोल | `/ɡol/` | `0.333` |
| **STOP** | स्लिप | `/slip/` | `0.500` |
| **STOP** | स्तर | `/stər/` | `0.500` |
| **UP** | अप | `/ap/` | `0.500` |
| **UP** | उप | `/up/` | `0.500` |
| **ON** | इन | `/in/` | `0.500` |
| **ON** | उन | `/un/` | `0.500` |
| **STOP** | सोच | `/sot͡ʃ/` | `0.600` |

### 2. Multi-Corpus Parallel Dataset Curation

To prevent acoustic overfitting to North American English or scripted studio Hindi, four corpora are streamed and standardized to $16\text{ kHz}$ mono PCM:

| Dataset | Role in Pipeline | Acoustic Coverage & Justification |
| --- | --- | --- |
| **[Google Speech Commands v2](https://arxiv.org/abs/1804.03209)** | English Positives (`Label 1`) & Noise (`Label 0`) | $19,065$ isolated 1-second target utterances + ambient background noise recordings. |
| **[AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)** | Indian-English Positives (`1`) & OOV Negatives (`0`) | Read & spontaneous English across 117 Indian speakers (19 states); prevents accent bias on wake-words while supplying non-keyword English negatives. |
| **[AI4Bharat Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath)** | Scripted Hindi Hard Negatives (`Label 0`) | Clean phonetic realizations of mined trap words sliced into verified 1.0s context windows. |
| **[AI4Bharat IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)** | Conversational Hindi Hard Negatives (`Label 0`) | $>93\%$ extempore natural speech capturing rapid co-articulation and regional Hindi dialects. |

### 3. Acoustic Frontend & Balanced Training

* **Log-Mel Spectrogram Extraction:** $16\text{ kHz}$ waveforms ($16,000$ samples) are transformed using a $25\text{ ms}$ Hann window (`n_fft=400`), $10\text{ ms}$ stride (`hop_length=160`), and $40$ Mel filterbanks, producing a compact input tensor of shape `[1, 40, 101]`.
* **Dynamic Class Balancing:** Because positive target keywords outnumber hard negatives by $>25:1$ ($96.3\%$ vs. $3.7\%$), a PyTorch `WeightedRandomSampler` enforces an exact **50% Positive / 50% Negative** split inside every training mini-batch.
* **Optimization:** Trained for 10 epochs using `AdamW` ($\text{lr} = 10^{-3}$, weight decay $= 10^{-2}$) with Cosine Annealing learning rate decay.

### 4. Edge Streaming Post-Processing (Smoothing + NMS)

Evaluating a continuous $16\text{ kHz}$ audio stream at a $20\text{ ms}$ hop ($50\text{ Hz}$ MCU buffer update rate) causes a single $400\text{ ms}$ confusable Hindi word to overlap $\sim 20$ consecutive sliding windows. Without post-processing, 2 spoken words inflate into $48$ frame-level false alarms ($1,310.42\text{ FA/hr}$).

Our edge inference wrapper applies two lightweight temporal filters:

1. **Posterior Moving Average:** Smooths class-1 softmax probabilities over a rolling **5-frame ($100\text{ ms}$) buffer** to reject transient spectral spikes.
2. **Non-Maximum Suppression (Debouncing):** Enforces a **25-frame ($500\text{ ms}$) refractory lockout** immediately following any trigger event above threshold $\tau = 0.85$.

---

## Comparative Model Benchmarks

All candidate architectures strictly adhere to the **$< 250\text{ KB}$ microcontroller SRAM budget** and accept identical `[Batch, 1, 40, 101]` Log-Mel inputs:

| Architecture | Design Paradigm | Parameters | FP32 RAM | ONNX Binary Size | Validation Acc. | Unfiltered FAR | Debounced FAR (NMS) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DS-CNN** *(Baseline)* | 2D Depthwise Separable Conv | `12,546` | `49.01 KB` | **`21.29 KB`** | `99.93%` | `1,310.42 FA/hr` | **`54.60 FA/hr`** *(2 events)* |
| **TC-ResNet8** | 1D Temporal Conv + Residual Skips | `33,450` | `130.66 KB` | `~132 KB` | *Benchmarking* | — | — |
| **BC-ResNet** | 2D Freq-DW + 1D Broadcasted Residual | `13,178` | `51.48 KB` | `~53 KB` | *Benchmarking* | — | — |

> **Key Finding on Quantization Overhead:** For ultra-compact networks under $50\text{ KB}$ ($\approx 12.5\text{k}$ parameters), dynamic INT8 ONNX quantization increases binary size from `21.29 KB` to `30.21 KB` due to injected `QuantizeLinear`/`DequantizeLinear` graph nodes exceeding weight compression savings. Exporting the native **FP32 ONNX (`21.29 KB`)** or using **TFLite Static INT8** is optimal for deployment.

---

## Repository Structure

```text
cross-lingual-edge-kws/
├── checkpoints/
│   ├── edge_kws_hnm.pth                 # Trained DS-CNN FP32 weights
│   ├── tc_resnet8.pth                   # Trained TC-ResNet8 weights
│   └── bc_resnet.pth                    # Trained BC-ResNet weights
├── exported_models/
│   └── edge_kws_fp32.onnx               # Production 21.29 KB ONNX graph (Opset 13)
├── scripts/
│   ├── 01_phonetic_mining.py            # IPA conversion & Levenshtein collision matrix
│   ├── 02_slice_and_verify.py           # GSC download + 1.0s Kathbath audio slicing
│   ├── 03_mine_parallel.py              # Streaming IndicVoices & Svarah parallel ingestion
│   ├── models.py                        # LightweightEdgeKWS (DS-CNN), TCResNet8, BCResNet
│   ├── train.py                         # Balanced training loop (WeightedRandomSampler)
│   ├── eval_stream.py                   # 50 Hz continuous streaming evaluator with NMS
│   └── export_onnx.py                   # ONNX graph export & footprint verification
├── hard_negative_seed_words.csv         # 37 mined Hindi-English phonetic collision pairs
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

Accept the CC-BY-4.0 dataset terms on [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) and [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), then log in via CLI:

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

# Step 4: Train & benchmark DS-CNN, TC-ResNet8, and BC-ResNet
python scripts/train.py

# Step 5: Evaluate continuous streaming False Alarm Rate (FA/hr) with NMS debouncing
python scripts/eval_stream.py

# Step 6: Export production ONNX binary (< 250 KB)
python scripts/export_onnx.py

```

---

## Hardware Deployment Notes

* **Microcontrollers with Hardware FPU (STM32F4/F7/H7, ESP32-S3, RP2350):** Deploy `exported_models/edge_kws_fp32.onnx` (`21.29 KB`) directly via **STM32 X-CUBE-AI**, **ESP-DL**, or **ONNX Runtime Mobile** with zero quantization degradation.
* **Integer-Only Microcontrollers (ARM Cortex-M0/M3):** Convert the PyTorch checkpoint to **TFLite Static INT8** using a 100-sample representative calibration set to fuse scale/zero-point constants without dynamic node overhead (`~14 KB` flash footprint).

---

## References

1. **DS-CNN:** Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). *Hello Edge: Keyword Spotting on Microcontrollers*. arXiv:1711.07128.
2. **TC-ResNet:** Choi, S., Seo, S., Shin, B., Byun, H., Kersner, M., Kim, B., Kim, D., & Ha, S. (2019). *Temporal Convolution for Real-time Keyword Spotting on Mobile Devices*. INTERSPEECH 2019.
3. **BC-ResNet:** Kim, B., Chang, S., Lee, J., & Sung, D. (2021). *Broadcasted Residual Learning for Efficient Keyword Spotting*. INTERSPEECH 2021.
4. **Google Speech Commands:** Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*. arXiv:1804.03209.
5. **Kathbath & IndicVoices:** Javed, T., et al. (2023/2024). *Towards Building ASR Systems for the Next Billion Users* & *IndicVoices: Towards Building an Inclusive Multilingual Speech Dataset for Indian Languages*. AI4Bharat.
6. **Svarah:** Javed, T., et al. (2023). *Svarah: Evaluating English ASR Systems on Indian Accents*. AI4Bharat.

```

```
