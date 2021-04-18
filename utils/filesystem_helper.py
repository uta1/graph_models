from lib_imports import *

from config import config
from utils.platform_based_params import folders_delim, workplace_dir


def _remove_extension(file_name):
    return file_name.split('.')[0]


def _get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def _create_folder_if_not_exists(folder):
    if os.path.exists(folder):
        return
    os.mkdir(folder)


def _actual_folder_name(mode):
    if mode == 'train':
        return config.TRAIN_FOLDER_NAME
    if mode == 'val':
        return config.VAL_FOLDER_NAME
    if mode == 'test':
        return config.TEST_FOLDER_NAME


# Paths of data

def src_folder_path():
    return workplace_dir() + config.SRC_FOLDER_NAME + folders_delim()

def trg_folder_path():
    return workplace_dir() + config.TRG_FOLDER_NAME + folders_delim()


# Name of src-file containing metainfo
# Available only for samples, train and val

def labels_file_name(mode):
    return _actual_folder_name(mode) + '.json'

def labels_file_path(mode):
    return src_folder_path() + labels_file_name(mode)


# Directory of dataset tagged by mode

def folder_path(mode):
    return src_folder_path() + _actual_folder_name(mode) + folders_delim()


# Directories of labels and target images

def bins_folder_path(mode):
    return trg_folder_path() + _actual_folder_name(mode) + '_bins' + folders_delim()

def labels_folder_path(mode):
    return trg_folder_path() + _actual_folder_name(mode) + '_labels' + folders_delim()

def jsons_folder_path(mode):
    return trg_folder_path() + _actual_folder_name(mode) + '_jsons' + folders_delim()


# Paths of image-related objects

def image_name_to_path(mode, image_name):
    return folder_path(mode) + image_name

def image_name_to_bin_path(mode, image_name):
    return bins_folder_path(mode) + 'trg_b_' + _remove_extension(image_name) + '.tiff'

def image_name_to_label_path(mode, image_name):
    return labels_folder_path(mode) + 'trg_l_' + _remove_extension(image_name) + '.tiff'


# Path of trg-file containing metainfo redesigned for our needs

def cached_labels_path(mode):
    return jsons_folder_path(mode) + 'cached_' + labels_file_name(mode)


# Logs properties

def logs_folder_path():
    return workplace_dir() + config.LOGS_FOLDER_NAME + folders_delim()

def log_file_path_snapshot():
    return logs_folder_path() + _log_file_name_snapshot()

def _log_file_name_snapshot():
    return config.LOG_FILE_NAME_TEMPLATE.format(
        time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()),
        config.MODEL
    )


# Network configs depended params

def weights_folder_path(network_config):
    return workplace_dir() + config.WEIGHTS_FOLDER_NAME_TEMPLATE.format(network_config.NAME) + folders_delim()

def weights_file_path_template(network_config):
    return weights_folder_path(network_config) + config.WEIGHTS_FILE_NAME_TEMPLATE


# Directories creation

def create_path(path):
    assert 'publaynet' not in path

    if path[-1] == '/':
        path = path[:-1]

    folders = path.split('/')

    cur_path = ''
    for folder in folders:
        cur_path += folder + '/'
        _create_folder_if_not_exists(cur_path)

def create_weights_folder(network_config):
    create_path(weights_folder_path(network_config))
