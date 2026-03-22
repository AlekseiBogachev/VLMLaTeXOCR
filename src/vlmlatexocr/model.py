def run_lora(
        model_config: str,
        lora_config: str,
        trainer_config: str,
        logging_config:str
    ):
    """Run SFT with LoRA.

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
    raise NotImplementedError("Comming soon.")


def run_all_weights_sft(
        model_config: str, trainer_config: str, logging_config: str
    ):
    """Run SFT SFT with optimization of all weights.

    Parameters
    ----------
    model_config : str
        Path to JSON with model parameters.
    trainer_config : str
        Path to JSON with Trainer parameters.
    logging_config : str
        Path to JSON with logging parameters.
    """
    raise NotImplementedError("Comming soon.")


def run_test():
    """Test model."""
    raise NotImplementedError("Comming soon.")

def run_predict(image_path: str, model_config: str, prediction_config: str):
    """Predict LaTeX for the image image_path.

    Parameters
    ----------
    image_path : str
        Path to the image to be recognized.
    model_config : str
        Path to JSON with model parameters.
    prediction_config : str
        Path to JSON with predictoin parameters.
    """
    raise NotImplementedError("Comming soon.")