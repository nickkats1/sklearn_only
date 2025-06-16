from src.common.utils import load_config,load_jobs,dump_jobs
from urllib.request import urlretrieve

def download_data(link,savepath):
    urlretrieve(link,savepath)

if __name__ == "__main__":
    config = load_config()
    savename = config["data_raw_dir"] + config["dataname"] + ".csv"
    download_data(config["datalink"],savename)