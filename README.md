# VLMLaTeXOCR
Vision-language model that converts handwritten formulas into LaTeX format.

## Overview
This project leverages Vision-Language Model (VLM), specifically `Qwen/Qwen3-VL-2B-Instruct`, to perform Optical Character Recognition (OCR) on handwritten mathematical formulas, converting them directly into LaTeX code.

## Features
- **Inference**: *Zero-shot* and *one-shot* inference to quickly test VLM capabilities on mathematical datasets.
- **Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT) using *LoRA*.
- **Metrics**: LaTeX text normalization paired with *Word Error Rate (WER)* and *Character Error Rate (CER)*.
- **Datasets**: `linxy/LaTeX_OCR` and `deepcopy/MathWriting-human` datasets and their combination.

## Project Structure
The source code is located in `src/vlmlatexocr/`:
- `data.py`: Utilities for downloading, caching, mixing datasets, and data_collator for fine-tuning.
- `model.py`: Functions for inference (`test_zero_shot_inference`, `test_one_shot_inference`) and fine-tuning (`run_lora`).
- `metrics.py`: Calculation of WER and CER metrics.

## Configuration
Model parameters, system prompts, data cache locations, and device configurations are managed via JSON files (e.g., `configs/model.json`).

## Command Line Interface (CLI)
The project includes CLI built with `click` (`src/vlmlatexocr/cli.py`) to easily run training and evaluation tasks from the terminal. Available commands include:
- `lora` - run Supervised Fine-Tuning (SFT) using LoRA;
- `test-zero-shot` - evaluate the model on a given dataset using zero-shot inference;
- `test-one-shot` - evaluate the model on a given dataset using one-shot inference.
