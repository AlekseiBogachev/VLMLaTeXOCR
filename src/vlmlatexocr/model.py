from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from vlmlatexocr.metrics import calculate_metrics
from vlmlatexocr.utils import read_config
from vlmlatexocr.data import get_dataset


def test_zero_shot_inference(
    dataset_name: str,
    model_config: str,
    num_samples: int | None = None,
    random_state: int = 42,
) -> dict:
    """Test zero-shot inference of the model on a specified dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to use for testing.
    model_config : str
        Path to the JSON configuration file containing model parameters.
    num_samples : int | None, optional
        The number of samples to test. If None, the entire test set is used,
        by default None.
    random_state : int, optional
        The random seed for shuffling the dataset, by default 42.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics (e.g., WER and CER).
    """
    model_params = read_config(model_config)

    processor = AutoProcessor.from_pretrained(
        model_params["model_name"],
        cache_dir=model_params["model_kwargs"]["cache_dir"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_params["model_name"],
        **model_params["model_kwargs"],
    )

    dataset = get_dataset(
        dataset_name,
        model_params["datasets_cache_dir"],
        random_state,
    )

    dataset = dataset["test"]

    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_state).select(range(num_samples))

    true_values = list()
    pred_values = list()

    for sample in tqdm(dataset):
        image = sample["image"]
        true_values.append(sample["text"])

        message_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": model_params["prompt_text"]},
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            message_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=model_params["max_new_tokens"],
            )

        generated_tokens = output_ids[:, prompt_len:]
        
        pred_values.append(
            processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        )

    return calculate_metrics(true_values, pred_values)


def test_one_shot_inference(
    dataset_name: str,
    model_config:str,
    num_samples: int | None = None,
    random_state: int = 42,
) -> dict:
    """Test one-shot inference of the model on a specified dataset.

    This function provides the model with one reference example from
    the dataset before asking it to predict the LaTeX code for the
    target image.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to use for testing.
    model_config : str
        Path to the JSON configuration file containing model parameters.
    num_samples : int | None, optional
        The number of samples to test. If None, the entire test set is used,
          by default None.
    random_state : int, optional
        The random seed for shuffling the dataset and selecting the reference
        example, by default 42.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics (e.g., WER and CER).
    """
    model_params = read_config(model_config)

    processor = AutoProcessor.from_pretrained(
        model_params["model_name"],
        cache_dir=model_params["model_kwargs"]["cache_dir"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_params["model_name"],
        **model_params["model_kwargs"],
    )

    dataset = get_dataset(
        dataset_name,
        model_params["datasets_cache_dir"],
        random_state,
    )

    dataset = dataset["test"]

    ref_example = dataset.shuffle(seed=random_state).select([0])
    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_state).select(range(num_samples))

    true_values = list()
    pred_values = list()



    for sample in tqdm(dataset):
        true_values.append(sample["text"])

        message_template = [
            {"role": "system",
             "content": [{"type": "text", "text": model_params["prompt_text"]}]
            },
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": "Convert this image to LaTeX:"}
             ]
            },
            {"role": "assistant",
             "content": [{"type": "text", "text": ref_example["text"]}]
            },
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": "Convert this image to LaTeX:"}
             ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            message_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text_prompt],
            images=[ref_example["image"], sample["image"]], 
            padding=True,
            return_tensors="pt"
        )

        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=model_params["max_new_tokens"],
            )

        generated_tokens = output_ids[:, prompt_len:]
        
        pred_values.append(
            processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        )

    return calculate_metrics(true_values, pred_values)


def run_lora(
    model_config: str,
    lora_config: str,
    trainer_config: str,
    logging_config: str,
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
