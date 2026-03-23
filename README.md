# VLMLaTeXOCR
Vision-language model that converts handwritten formulas into LaTeX format.

## Overview
This project leverages Vision-Language Models (VLMs), specifically models like `Qwen/Qwen3-VL-2B-Instruct`, to perform Optical Character Recognition (OCR) on handwritten mathematical formulas, converting them directly into valid LaTeX code.

## Features
- **Inference**: Support for zero-shot and one-shot inference to quickly test VLM capabilities on mathematical datasets.
- **Fine-Tuning**: Built-in support for Parameter-Efficient Fine-Tuning (PEFT) using LoRA via Hugging Face's `Trainer`.
- **Metrics**: Custom LaTeX text normalization paired with Word Error Rate (WER) and Character Error Rate (CER) calculation using the `evaluate` library.
- **Datasets**: Built-in data loaders and mixers for the `linxy/LaTeX_OCR` and `deepcopy/MathWriting-human` datasets.

## Project Structure
The core source code is located in `src/vlmlatexocr/`:
- `data.py`: Utilities for downloading, caching, mixing datasets, and a custom `VLMDataCollator` for formatting chat templates and image inputs.
- `model.py`: Main execution functions for inference (`test_zero_shot_inference`, `test_one_shot_inference`) and fine-tuning (`run_lora`).
- `metrics.py`: Calculation of WER and CER metrics, featuring a `normalize_latex` string cleaner to ensure meaningful and accurate evaluations.

## Configuration
Model parameters, system prompts, data cache locations, and device configurations are managed via JSON files (e.g., `configs/model.json`).

## Command Line Interface (CLI)
The project includes a robust CLI built with `click` (`src/vlmlatexocr/cli.py`) to easily run training and evaluation tasks from the terminal. Available commands include:
- `lora`: Run Supervised Fine-Tuning (SFT) using LoRA.
- `test-zero-shot`: Evaluate the model on a given dataset using zero-shot inference.
- `test-one-shot`: Evaluate the model on a given dataset using one-shot inference.
- `predict`: Extract and predict the LaTeX code from a single target image path.

## Dependencies
Built using standard Machine Learning ecosystems including `torch`, `transformers`, `peft`, `datasets`, and `evaluate`. Project dependencies are managed via Poetry.
