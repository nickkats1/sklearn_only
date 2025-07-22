import pytest
import pandas as pd
from src.data_processing.data_transformation import DataTransformation
from src.config import load_config

@pytest.fixture()
def dummy_data_transformation():
    config = load_config()
    df = pd.DataFrame({
        'col_a': [1, 2, 4, 4],
        'col_b': ['a', 'b', 'c', 'd'],
        'col_c': [10.0, 20.0, 30.0, 40.0]
    })
    dummy_transformation = DataTransformation(config)
    return dummy_transformation.transform_data()

@pytest.mark.parametrize("input_value, expected_output", [
    (5, 10),
    (10, 20),
    (0, 0)
])
def test_some_other_transformation_logic(dummy_data_transformation, input_value, expected_output):
    transformed_value = dummy_data_transformation.some_other_transformation_method(input_value)
    assert transformed_value == expected_output
