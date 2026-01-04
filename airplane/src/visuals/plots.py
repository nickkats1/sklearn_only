# data transformation
from src.data.data_transformation import DataTransformation
from src.data.data_ingestion import DataIngestion
from helpers.config import load_config
from helpers.logger import logger

# import plots
import matplotlib.pyplot as plt
import seaborn as sns

# pandas
import pandas as pd

class Plots:
    """Plot data from data ingestion."""
    
    def __init__(self, config: dict, data: DataIngestion | None = None):
        """Initialize plots.
        
        Args:
            config (dict): A configuration file consisting of all folder paths, features, targets, etc.
        """
        
        self.config = config or load_config()
        self.data =  data or DataIngestion(self.config).fetch_data()
        
    def heatmap(self) -> None:
        """Return plots from data ingestion"""
        
        try:
            
            # drop duplicates
            self.data.drop_duplicates(inplace=True)
            
            plt.figure(figsize=(10,6))
            sns.heatmap(self.data.corr(), annot=True, cmap="Blues", fmt=".2f")
            plt.title("Heatmap of features")
            plt.tight_layout()
            plt.show()
        except ValueError as e:
            logger.error(f"Could not plot heatmap: {e}")
        return None
    
    def plot_descriptive_stats(self) -> None:
        """Plots of variables describing the data."""
        
        try:
            fig, axs = plt.subplots(2,2, figsize=(10,5))
            
            sns.lineplot(x="age", y="wtop", label="Age compared to wtop", data=self.data, ax=axs[0,0])
            axs[0,0].set_title("Age compared to wtop")
            axs[0,0].legend()
            
            sns.scatterplot(x="fixgear", y="price", label="Fix Gear Compared to price", data=self.data, ax=axs[0,1])
            axs[0,1].set_title("Fix Gear Compared to price")
            axs[0,1].legend()
            
            sns.boxplot(x="horse", y="wtop", hue="price", data=self.data, ax=axs[1,0])
            axs[1,0].set_title("Horse power compared to wtop compared to price")
            axs[1,0].legend()
            
            sns.lineplot(x="tdrag", y="price", hue="age", data=self.data, ax=axs[1,1])
            axs[1,1].set_title("trag compared to price based on age")
            axs[1,1].tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Could not perform descriptive stats based on value error: {e}")
        return None