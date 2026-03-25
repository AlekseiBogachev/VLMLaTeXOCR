from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def sample_val_test(dataset: Dataset, val_frac=0.1, test_frac=0.1, seed=42):
    """Split a dataset into train, validation, and test sets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    val_frac : float, optional
        The fraction of the dataset to include in the validation split,
        by default 0.1.
    test_frac : float, optional
        The fraction of the dataset to include in the test split,
        by default 0.1.
    seed : int, optional
        Random seed for shuffling the dataset before splitting,
        by default 42.

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        A tuple containing the train, validation, and test datasets.
    """
    data = dataset.shuffle(seed=seed)
    val_size = int(len(data) * val_frac)
    test_size = int(len(data) * test_frac)
    train_size = len(data) - val_size - test_size

    train_boundaries = (0, train_size)
    val_boundaries = (train_size, train_size + val_size)
    test_start = train_size + val_size
    test_boundaries = (test_start, test_start + test_size)

    return (
        data.select(range(*train_boundaries)),
        data.select(range(*val_boundaries)),
        data.select(range(*test_boundaries)),
    )


def get_latex_ocr(cache_dir: Path) -> Dataset:
    """Load linxy/LaTeX_OCR dataset.

    This function loads the linxy/LaTeX_OCR dataset, which contains
    images of LaTeX formulas and their corresponding LaTeX source code.
    It specifically loads the 'full' configuration of the dataset.

    Parameters
    ----------
    cache_dir : Path
        The directory where the dataset will be cached locally.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object containing the LaTeX OCR data.
    """
    data = load_dataset(
        "linxy/LaTeX_OCR",
        "default",
        split="train",
        cache_dir=cache_dir,
    )
    train, val, test = sample_val_test(data)

    return DatasetDict(
        {
            "train": train,
            "validation": val,
            "test": test,
        }
    )


def get_mathwriting(cache_dir: Path) -> Dataset:
    """Load deepcopy/MathWriting-human dataset.

    This function loads the deepcopy/MathWriting-human dataset, which
    contains images of handwritten mathematical expressions and their
    LaTeX representations. It renames the 'latex' column to 'text'
    and selects only the 'image' and 'text' columns to match a common
    format.

    Parameters
    ----------
    cache_dir : Path
        The directory where the dataset will be cached locally.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object containing the preprocessed
        MathWriting data.
    """
    data = (
        load_dataset(
            "deepcopy/MathWriting-human",
            split="train",
            cache_dir=cache_dir,
        )
        .rename_column("latex", "text")
        .select_columns(["image", "text"])
    )

    train, val, test = sample_val_test(data)

    return DatasetDict(
        {
            "train": train,
            "validation": val,
            "test": test,
        }
    )


def get_mixed_dataset(cache_dir: Path, random_state: int = 42) -> Dataset:
    """Create a dataset by concatenating LaTeX OCR and MathWriting datasets.

    This function combines the linxy/LaTeX_OCR and deepcopy/MathWriting-human
    datasets into a single dataset. After concatenation, the combined dataset
    is shuffled to ensure randomness.

    Parameters
    ----------
    cache_dir : Path
        The directory where the individual datasets will be cached locally.
    random_state : int, optional
        Seed for shuffling the combined dataset, by default 42.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object containing the mixed and shuffled data.
    """
    latex_ocr = get_latex_ocr(cache_dir)
    mathwriting = get_mathwriting(cache_dir)

    return DatasetDict(
        {
            key: concatenate_datasets(
                [latex_ocr[key], mathwriting[key]]
            ).shuffle(seed=random_state)
            for key in latex_ocr
        }
    )


def get_dataset(
    dataset_name: str, cache_dir: Path, random_state: int = 42
) -> Dataset:
    """Check dataset_name and return loaded dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name. Allowed values are 'latex_ocr',
        'mathwriting', or 'mixed'.
    cache_dir : Path
        The directory where the dataset will be cached locally.
    random_state : int, optional
        Seed for shuffling the combined dataset, by default 42.

    Returns
    -------
    Dataset
        Loaded dataset.

    Raises
    ------
    ValueError
        Raise ValueError if dataset_name is not one of the allowed values.
    """
    if dataset_name == "latex_ocr":
        dataset = get_latex_ocr(cache_dir)
    elif dataset_name == "mathwriting":
        dataset = get_mathwriting(cache_dir)
    elif dataset_name == "mixed":
        dataset = get_mixed_dataset(cache_dir, random_state)
    else:
        raise ValueError(
            "Dataset must be one of 'latex_ocr', 'mathwriting', or 'mixed'."
        )

    return dataset


class VLMDataCollator:
    def __init__(self, processor, system_prompt: str):
        self.processor = processor
        self.system_prompt = system_prompt

    def __call__(
        self, samples: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        texts = []
        images = []

        for sample in samples:
            message_template = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Convert this image to LaTeX:",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]

            text = self.processor.apply_chat_template(
                message_template, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(sample["image"])

        batch = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )

        labels = batch["input_ids"].clone()

        # Convert prompt ending (separator) to tokens
        separator = "<|im_start|>assistant\n"
        separator_ids = self.processor.tokenizer(
            separator, add_special_tokens=False
        )["input_ids"]
        
        # Find separator's tokens and mask prompt
        separator_len = len(separator_ids)
        for seq in batch:
            for i in range(len(seq)):
                if torch.equal(seq[i:i+separator_len], torch.tensor(separator_ids)):
                    seq[:i] = -100
                    break

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch
