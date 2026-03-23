from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


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
    return load_dataset(
        "linxy/LaTeX_OCR",
        "full",
        cache_dir=cache_dir,
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
    return (
        load_dataset(
            "deepcopy/MathWriting-human",
            "default",
            cache_dir=cache_dir,
        )
        .rename_column("latex", "text")
        .select_columns(["image", "text"])
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
    mathwriting = DatasetDict(
        {
            "train": mathwriting["train"],
            "validation": mathwriting["val"],
            "test": mathwriting["test"],
        }
    )

    return DatasetDict(
        {
            key: concatenate_datasets(
                [latex_ocr[key], mathwriting[key]]
            ).shuffle(seed=random_state)
            for key in latex_ocr
        }
    )
