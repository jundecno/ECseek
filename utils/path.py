import rootutils
root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))

from tqdm import tqdm

def get_dataset_path():
    return root_path + "/datasets/"


def get_tools_path():
    return root_path + "/tools/"


def get_configs_path():
    return root_path + "/configs/"


def get_info_path():
    return get_dataset_path() + "/_info/"


def get_statis_path():
    return get_dataset_path() + "/_info/statis/"


def get_labels_path():
    return get_dataset_path() + "/_info/labels/"


def get_feature_path():
    return get_dataset_path() + "/_feat/"

data_path = get_dataset_path()
tools_path = get_tools_path()
info_path = get_info_path()
configs_path = get_configs_path()
statis_path = get_statis_path()
labels_path = get_labels_path()
feature_path = get_feature_path()
global_tools_path = "xxx/tools/"
