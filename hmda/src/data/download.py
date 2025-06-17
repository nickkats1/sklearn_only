from urllib.request import urlretrieve
import src.common.utils as tools

def download_data(link,savepath):
    urlretrieve(link,savepath)

if __name__ == "__main__":
    config = tools.load_config()
    savename = config["raw_path"] + config["data_name"] + ".csv"
    download_data(config["url_name"],savename)