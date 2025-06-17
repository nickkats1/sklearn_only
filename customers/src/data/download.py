from src.common.utils import load_config
from urllib.request import urlretrieve


def download_data(link,savepath):
    urlretrieve(link,savepath)


if __name__ == "__main__":
    config = load_config()
    savename = config["dataraw_dir"] + config["dataname"] + ".csv"
    download_data(config["datalink"],savename)