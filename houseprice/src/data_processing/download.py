import src.common.utils as tools
from urllib.request import urlretrieve

def download_data(link,savepath):
    urlretrieve(link,savepath)

if __name__ == "__main__":
    config = tools.load_config()
    savename = config["raw_data_file"] + config["package_name"] + ".csv"
    download_data(config["datalink"],savename)