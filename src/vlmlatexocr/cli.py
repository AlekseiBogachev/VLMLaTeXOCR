from pathlib import Path
from pprint import pprint

import click

from vlmlatexocr.model import (
    run_all_weights_sft,
    run_lora,
    run_predict,
    test_one_shot_inference,
    test_zero_shot_inference,
)
from vlmlatexocr.service import start_sevice


@click.command()
@click.argument(
    "dataset-name",
    type=str,
)
@click.option(
    "--model-config",
    default=Path("./configs/model.json"),
    help="Path to JSON with model parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--lora-config",
    default=Path("./configs/lora.json"),
    help="Path to JSON with LoRA parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--trainer-config",
    default=Path("./configs/trainer.json"),
    help="Path to JSON with Trainer parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--frac",
    help="Fraction of the dataset to use for SFT",
    type=float,
)
@click.option(
    "--random-state",
    default=42,
    help="Random state for shuffling the dataset",
    show_default=True,
    type=int,
)
def lora(
    dataset_name: str,
    model_config: str,
    lora_config: str,
    trainer_config: str,
    frac: float | None = None,
    random_state: int = 42,
):
    """Run SFT with LoRA.\f

    Parameters
    ----------
    dataset_name : str
        Dataset name. Allowed values are 'latex_ocr', 'mathwriting',
        or 'mixed'.
    model_config : str
        Path to JSON with model parameters.
    lora_config : str
        Path to JSON with LoRA parameters.
    trainer_config : str
        Path to SON with Trainer parameters.
    logging_config : str
        Path to JSON with logging parameters.
    frac : float | None, optional
        Fraction of the dataset to use for SFT. If None, the entire
        dataset is used. Default None.
    random_state : int, optional
        Random state for shuffling the dataset and selecting the reference
        example, by default 42.
    """
    run_lora(
        dataset_name,
        model_config,
        lora_config,
        trainer_config,
        random_state,
    )


@click.command()
@click.option(
    "--model-config",
    default=Path("./configs/model.json"),
    help="Path to JSON with model parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--trainer-config",
    default=Path("./configs/trainer.json"),
    help="Path to JSON with Trainer parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--logging-config",
    default=Path("./configs/logging.json"),
    help="Path to JSON with logging parameters",
    show_default=True,
    type=click.Path(exists=True),
)
def fine_tune_weights(
    model_config: str, trainer_config: str, logging_config: str
):
    """Run SFT with optimization of all weights.\f

    Parameters
    ----------
    model_config : str
        Path to JSON with model parameters
    trainer_config : str
        Path to JSON with Trainer parameters
    logging_config : str
        Path to JSON with logging parameters
    """
    run_all_weights_sft(model_config, trainer_config, logging_config)


@click.command()
@click.argument(
    "dataset-name",
    type=str,
)
@click.option(
    "--model-config",
    default=Path("./configs/model.json"),
    help="Path to JSON with model parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--num-samples",
    help="Number of samples to use for testing",
    type=int,
)
@click.option(
    "--random-state",
    default=42,
    help="Random state for shuffling the dataset",
    show_default=True,
    type=int,
)
def test_zero_shot(
    dataset_name: str,
    model_config: str,
    num_samples: int | None = None,
    random_state: int = 42,
):
    """Test zero-shot inferens on dataset dataset_name with model_config.

    Allowed values for dataset_name are 'latex_ocr', 'mathwriting', or
    'mixed'.\f

    Parameters
    ----------
    dataset_name : str
        Dataset name. Allowed values are 'latex_ocr', 'mathwriting',
        or 'mixed'.
    model_config : str
        Path to JSON with model parameters.
    num_samples : int | None
        Number of samples to use for testing. If None, the entire
        test set is used.
    random_state : int, optional
        Random state for shuffling the dataset, by default 42.
    """
    print(f"Test zero-shot inference on {dataset_name} dataset.")
    pprint(
        test_zero_shot_inference(
            dataset_name, model_config, num_samples, random_state
        )
    )


@click.command()
@click.argument(
    "dataset-name",
    type=str,
)
@click.option(
    "--model-config",
    default=Path("./configs/model.json"),
    help="Path to JSON with model parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--num-samples",
    help="Number of samples to use for testing",
    type=int,
)
@click.option(
    "--random-state",
    default=42,
    help="Random state for shuffling the dataset",
    show_default=True,
    type=int,
)
def test_one_shot(
    dataset_name: str,
    model_config: str,
    num_samples: int | None = None,
    random_state: int = 42,
):
    """Test one-shot inferens on dataset dataset_name with model_config.

    Allowed values for dataset_name are 'latex_ocr', 'mathwriting', or
    'mixed'.\f

    Parameters
    ----------
    dataset_name : str
        Dataset name. Allowed values are 'latex_ocr', 'mathwriting', or 'mixed'.
    model_config : str
        Path to JSON with model parameters.
    num_samples : int | None
        Number of samples to use for testing. If None, the entire
        test set is used.
    random_state : int, optional
        Random state for shuffling the dataset, by default 42.
    """
    print(f"Test one-shot inference on {dataset_name} dataset.")
    pprint(
        test_one_shot_inference(
            dataset_name, model_config, num_samples, random_state
        )
    )


@click.command()
@click.argument(
    "image-path",
    type=click.Path(exists=True),
)
@click.option(
    "--model-config",
    default=Path("./configs/model.json"),
    help="Path to JSON with model parameters",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--prediction-config",
    default=Path("./configs/predict.json"),
    help="Path to JSON with predictoin parameters",
    show_default=True,
    type=click.Path(exists=True),
)
def predict(image_path: str, model_config: str, prediction_config: str):
    """Predict LaTeX for the image image_path.\f

    Parameters
    ----------
    model_config : str
        Path to JSON with model parameters
    image_path : str
        Path to the image to be recognized.
    prediction_config : str
        Path to JSON with predictoin parameters.
    """
    run_predict(image_path, model_config, prediction_config)


@click.command()
def run_service():
    start_sevice()


@click.group()
def cli():
    """Run VLMLaTeXOCR CLI."""
    pass


cli.add_command(lora)
# cli.add_command(fine_tune_weights)
cli.add_command(test_zero_shot)
cli.add_command(test_one_shot)
cli.add_command(predict)
# cli.add_command(run_service)


if __name__ == "__main__":
    cli()
