#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

poetry run python cli.py test-zero-shot latex_ocr --num-samples 100 2>&1 | tee logs/test_zero_shot_${TIMESTAMP}.log
poetry run python cli.py test-one-shot latex_ocr --num-samples 100 2>&1 | tee logs/test_one_shot_${TIMESTAMP}.log
poetry run python run_lora.py --frac 0.01 2>&1 | tee logs/run_lora_${TIMESTAMP}.log
