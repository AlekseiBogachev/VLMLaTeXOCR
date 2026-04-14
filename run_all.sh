#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

poetry run python src/vlmlatexocr/cli.py test-zero-shot latex_ocr --num-samples 200 2>&1 | tee logs/test_zero_shot_latex_ocr_${TIMESTAMP}.log
poetry run python src/vlmlatexocr/cli.py test-one-shot latex_ocr --num-samples 200 2>&1 | tee logs/test_one_shot_latex_ocr_${TIMESTAMP}.log

poetry run python src/vlmlatexocr/cli.py test-zero-shot mathwriting --num-samples 200 2>&1 | tee logs/test_zero_shot_mathwriting_${TIMESTAMP}.log
poetry run python src/vlmlatexocr/cli.py test-one-shot mathwriting --num-samples 200 2>&1 | tee logs/test_one_shot_mathwriting_${TIMESTAMP}.log

poetry run python src/vlmlatexocr/cli.py test-zero-shot mixed --num-samples 200 2>&1 | tee logs/test_zero_shot_mixed_${TIMESTAMP}.log
poetry run python src/vlmlatexocr/cli.py test-one-shot mixed --num-samples 200 2>&1 | tee logs/test_one_shot_mixed_${TIMESTAMP}.log

poetry run python src/vlmlatexocr/cli.py lora latex_ocr --frac 0.1 2>&1 | tee logs/run_lora_latex_ocr_${TIMESTAMP}.log
