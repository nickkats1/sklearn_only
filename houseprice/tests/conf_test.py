import pytest
import pandas as pd
from src.config import load_config


@pytest.fixture
def dummy_input_data():
    config = load_config()
    data_path = config['data_raw']
    return pd.read_csv(data_path,delimiter=",")


