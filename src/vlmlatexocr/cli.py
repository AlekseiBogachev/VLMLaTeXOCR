from pathlib import Path

import click

from vlmlatexocr.service import start_sevice
from vlmlatexocr.train import (
    run_lora,
    run_all_weights_sft,
    run_test,
    run_predict,
)


@click.command()
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
def lora(lora_config: str, trainer_config: str, logging_config:str):
    """Run SFT with LoRA.\f

    
    Parameters
    ----------
    lora_config : str
        Path to JSON with LoRA parameters.
    trainer_config : str
        Path to SON with Trainer parameters.
    logging_config : str
        Path to JSON with logging parameters.
    """
    run_lora(lora_config, trainer_config, logging_config)


@click.command()
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
def fine_tune_weights(trainer_config: str, logging_config:str):
    """Run SFT SFT with optimization of all weights.\f

    Parameters
    ----------
    trainer_config : str
        Path to JSON with Trainer parameters
    logging_config : str
        Path to JSON with logging parameters
    """
    run_all_weights_sft(trainer_config, logging_config)


@click.command()
def test():
    run_test()


@click.command()
@click.argument(
    "image-path",
    type=click.Path(exists=True),
)
@click.option(
    "--config",
    default=Path("./configs/predict.json"),
    help="Path to JSON with predictoin parameters",
    show_default=True,
    type=click.Path(exists=True),
)
def predict(image_path: str, config: str):
    """Predict LaTeX for the image image_path.\f

    Parameters
    ----------
    image_path : str
        Path to the image to be recognized.
    config : str
        Path to JSON with predictoin parameters.
    """
    run_predict(image_path, config)


@click.command()
def run_service():
    start_sevice()


@click.group()
def cli():
    pass


cli.add_command(lora)
cli.add_command(fine_tune_weights)
cli.add_command(test)
cli.add_command(predict)
cli.add_command(run_service)


if __name__ == "__main__":
    cli()
