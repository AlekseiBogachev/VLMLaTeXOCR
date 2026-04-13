# VLMLaTeXOCR
Vision-language model that converts handwritten formulas into LaTeX format.

## Overview
This project leverages a Vision-Language Model (VLM), specifically `Qwen/Qwen3-VL-2B-Instruct`, to perform Optical Character Recognition (OCR) on handwritten mathematical formulas, converting them directly into LaTeX code.

## Features
- **Inference**: *Zero-shot* and *one-shot* inference to quickly test VLM capabilities on mathematical datasets.
- **Fine-Tuning**: Parameter-Efficient Fine-Tuning (PEFT) using *LoRA*.
- **Metrics**: *Word Error Rate (WER)* and *Character Error Rate (CER)* on
normalized LaTeX.
- **Datasets**: `linxy/LaTeX_OCR` and `deepcopy/MathWriting-human` datasets and their combination.
- **Prediction**: Predict LaTeX for the **.jpg** with an equation using
one-shot inference.

## Project Structure
The source code is located in `src/vlmlatexocr/`:
- `data.py`: Utilities for downloading, caching, mixing datasets, and data_collator for fine-tuning.
- `model.py`: Functions for inference (`test_zero_shot_inference`, `test_one_shot_inference`) and fine-tuning (`run_lora`).
- `metrics.py`: Calculation of WER and CER metrics.
- `cli.py`: Command Line Interface built with `click`.

## Configuration
Model parameters, system prompts, data cache locations, and device configurations are managed via JSON files (e.g., `configs/model.json`).

## Usage
The project includes a Command Line Interface (CLI) built with `click`.

### Fine-tuning with LoRA
To run Supervised Fine-Tuning (SFT) using LoRA:
```bash
poetry run python src/vlmlatexocr/cli.py lora latex_ocr --frac 0.1
```

### Zero-shot inference evaluation
To evaluate the base model on a dataset using zero-shot inference:
```bash
poetry run python src/vlmlatexocr/cli.py test-zero-shot latex_ocr --num-samples 200
```

### One-shot inference evaluation
To evaluate the model using one-shot inference, providing one example in the prompt:
```bash
poetry run python src/vlmlatexocr/cli.py test-one-shot latex_ocr --num-samples 200
```

### Prediction with one-shot inference
To predict LaTeX for the **.jpg** with an equation using one-shot inference:
```bash
poetry run python src/vlmlatexocr/cli.py predict equation_example.jpg
```