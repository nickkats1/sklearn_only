"""Data ingestion module.

This module provides functionality for retrieving data from a configured
URL source and returning it as a pandas DataFrame.
"""

# import pandas as pd
import pandas as pd

# config
from helpers.config import load_config
from helpers.logger import logger
# numpy
import numpy as np

from typing import Optional, Dict
class DataIngestion:
    """Responsible for raw data ingestion."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Initialize the data ingestion pipeline.

        Args:
            config (Optional[Dict]): Configuration dictionary containing URL
                links and file paths. If not provided, the default configuration
                is loaded.
        """
        self.config = config or load_config()

    def fetch_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch raw airplane data from configured URL sources.

        Returns:
            Optional[pd.DataFrame]: Combined raw dataset if successful,
            otherwise None.
        """
        try:
            sales_link = self.config["airplane_sales_link"]
            specs_link = self.config["airplane_specs_link"]
            perf_link = self.config["airplane_perf_link"]

            airplane_sales = pd.read_csv(sales_link)
            airplane_specs = pd.read_csv(specs_link)
            airplane_perf = pd.read_csv(perf_link)

            logger.info("Shape of airplane sales: %s", airplane_sales.shape)
            logger.info("Shape of airplane specs: %s", airplane_specs.shape)
            logger.info("Shape of airplane performance: %s", airplane_perf.shape)

            data = pd.concat(
                [airplane_sales, airplane_specs, airplane_perf],
                axis=1,
            )
            data = data.loc[:, ~data.columns.duplicated()].copy()

            logger.info("Shape of combined raw dataset: %s", data.shape)

            return data

        except FileNotFoundError as exc:
            logger.error("Data source could not be found: %s", exc)
            return None

