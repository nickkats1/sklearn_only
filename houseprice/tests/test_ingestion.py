import pytest
from src.data_processing.data_ingestion import DataIngestion 
from src.config import load_config

@pytest.fixture()
def dummy_data_ingestion():
    config = load_config()
    data_ingestion = DataIngestion(config)
    return data_ingestion.load_data()



def test_data_fetching(dummy_data_ingestion):
    assert dummy_data_ingestion is not None
    assert len(dummy_data_ingestion) > 0