from lib_imports import *

from config import config


def _remove_extension(file_name):
    return file_name.split('.')[0]


def _get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def _create_folder_if_not_exists(folder):
    if os.path.exists(folder):
        return
    os.mkdir(folder)


def image_name_to_path(mode, image_name):
    return config.folder_path(mode) + image_name


def image_name_to_bin_path(mode, image_name):
    return config.bins_folder_path(mode) + 'trg_b_' + _remove_extension(image_name) + '.tiff'


def image_name_to_label_path(mode, image_name):
    return config.labels_folder_path(mode) + 'trg_l_' + _remove_extension(image_name) + '.tiff'


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
    create_path(network_config.WEIGHTS_FOLDER_PATH)
