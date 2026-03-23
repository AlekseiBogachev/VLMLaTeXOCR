from pathlib import Path
from pprint import pprint

import click

from vlmlatexocr.model import (
    run_all_weights_sft,
    run_lora,
    run_predict,
    test_zero_shot_inference,
)
from vlmlatexocr.service import start_sevice


@click.command()
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
    "--logging-config",
    default=Path("./configs/logging.json"),
    help="Path to JSON with logging parameters",
    show_default=True,
    type=click.Path(exists=True),
)
def lora(
    model_config: str,
    lora_config: str,
    trainer_config: str,
    logging_config: str,
):
    """Run SFT with LoRA.\f

    Parameters
    ----------
    model_config : str
        Path to JSON with model parameters.
    lora_config : str
        Path to JSON with LoRA parameters.
    trainer_config : str
        Path to SON with Trainer parameters.
    logging_config : str
        Path to JSON with logging parameters.
    """
    run_lora(model_config, lora_config, trainer_config, logging_config)


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
def test_zero_shot(dataset_name: str, model_config:str):
    """Test zero-shot inferens on dataset dataset_name with model_config.\f

    Parameters
    ----------
    dataset_name : str
        Dataset name. Allowed values are 'latex_ocr', 'mathwriting', or 'mixed'.
    model_config : str
        Path to JSON with model parameters.
    """
    print(f"Test zero-shot inference on {dataset_name} dataset.")
    pprint(test_zero_shot_inference(dataset_name, model_config))


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
    """Run CLI."""
    pass


cli.add_command(lora)
# cli.add_command(fine_tune_weights)
cli.add_command(test_zero_shot)
cli.add_command(predict)
# cli.add_command(run_service)


if __name__ == "__main__":
    cli()
