# AI-text-detection

# Who's AI? Contrastive Learning for AI-Generated Text Detection & Attribution

This repository contains a PyTorch implementation of the paper **"[Is Contrasting All You Need?](https://arxiv.org/abs/2407.09364) Contrastive Learning for the Detection and Attribution of AI-Generated Text"**, developed from scratch. The model performs:

- **Binary Classification**: Distinguish between human-written and AI-generated text.
- **Author Attribution**: If generated, identify _which AI model_ (e.g., GPT-2, GPT-3) produced the text.

## Overview

Using contrastive learning principles, the model builds embeddings that are both discriminative for AI vs. Human detection and meaningful for AI author attribution. An ensemble of a supervised classifier and centroid-based matching is used during inference to enhance performance.

## Features

- Contrastive loss with **triplet-based InfoNCE**
- **BERT-based encoder** with a projection head
- **Author attribution** using source centroids
- Lightweight text augmentation via word dropout
- Designed for extensibility with new LLMs

## Dataset

We use a **subset of [TuringBench](https://github.com/TuringBench/TuringBench)**, filtered for the following classes:

- `human`
- `gpt1`, `gpt2_small`, `gpt2_medium`, `gpt2_large`, `gpt2_xl`, `gpt3`

Each instance includes:

- `Generation` (text)
- `label` (author/source)

## Model Architecture

- `BERT` encoder from HuggingFace Transformers
- Projection head (for contrastive representation)
- Classification head (for binary classification)
- Contrastive dataset with hard negative mining
- Dual loss: `Contrastive (InfoNCE + triplet)` + `CrossEntropy`

## Results

Evaluation was conducted using a subset of the [TuringBench](https://github.com/TuringBench/TuringBench) dataset across the following models:  
`gpt1`, `gpt2_small`, `gpt2_medium`, `gpt2_large`, `gpt2_xl`, `gpt3`, and `human`.

### 🔎 Task 1: Turing Test (Human vs AI Detection)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9584 |
| Precision | 0.9025 |
| Recall    | 0.9994 |
| F1-Score  | 0.9482 |

---

### 🧠 Task 2: Author Attribution (Identifying AI model)

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.9470 |
| Recall    | 0.9096 |
| F1-Score  | 0.9280 |
