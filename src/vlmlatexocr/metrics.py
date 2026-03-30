import re

import numpy as np
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def calculate_wer_cer(true: list[str], pred: list[str]) -> dict:
    """Calculate Word Error Rate (WER) and Character Error Rate (CER).

    Parameters
    ----------
    true : list[str]
        The ground truth LaTeX strings.
    pred : list[str]
        The predicted LaTeX strings.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics with keys 'wer'
        and 'cer'.
    """
    wer_score = wer_metric.compute(predictions=pred, references=true)
    cer_score = cer_metric.compute(predictions=pred, references=true)

    return {"wer": wer_score, "cer": cer_score}


def normalize_latex(text: str) -> str:
    """Normalize a LaTeX string for metric calculation.

    This function replaces any whitespace with a single space, separates
    structural characters and operators with spaces, detaches LaTeX commands
    from the following text, and strips redundant spaces. This ensures that
    metrics like Word Error Rate (WER) compute differences correctly on
    meaningful LaTeX tokens.

    Parameters
    ----------
    text : str
        The raw LaTeX string.

    Returns
    -------
    str
        The normalized LaTeX string.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([\{\}\^_=\+\-\(\)\[\]])", r" \1 ", text)
    text = re.sub(r"(\\[a-zA-Z]+)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_metrics(eval_res, processor=None) -> dict:
    """Normalize LaTeX strings and calculate WER and CER."""
    pred, true = eval_res

    if processor:
        # replace -100 with the pad_token_id for decoding
        true = np.where(true != -100, true, processor.tokenizer.pad_token_id)

        pred = processor.batch_decode(
            pred,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        true = processor.batch_decode(
            true,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return calculate_wer_cer(
        list(map(normalize_latex, true)),
        list(map(normalize_latex, pred)),
    )
