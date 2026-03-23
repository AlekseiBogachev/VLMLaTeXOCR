import json


def read_config(config_path: str) -> dict:
    """Read configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration data.
    """
    with open(config_path, "r") as f:
        return json.load(f)
