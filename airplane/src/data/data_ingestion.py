# import pandas as pd
import pandas as pd

# Config logger
from helpers.config import load_config
from helpers.logger import logger



class DataIngestion:
    """Responsible for raw data ingestion."""

    def __init__(self, config: dict):
        """
        Initialize the data ingestion pipeline.

        Args:
            config(dict): A configuration file with features, links, targets ect.
        """
        self.config = config or load_config()

    def fetch_raw_data(self) -> pd.DataFrame:
        """
        Fetch raw airplane data from configured URL sources.

        Returns:
            pd.DataFrame: Combined raw dataset if successful.
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

        except Exception as exc:
            logger.error(f"failed to retrieve file from link: {exc}")
            return None
