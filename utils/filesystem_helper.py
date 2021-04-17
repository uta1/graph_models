from lib_imports import *

from config import *


def remove_extension(file_name):
    return file_name.split('.')[0]


def image_name_to_path(image_name):
    return config.FOLDER + image_name


def image_name_to_bin_path(image_name):
    return config.BINS_FOLDER + 'trg_b_' + remove_extension(image_name) + '.tiff'


def image_name_to_json_path(image_name):
    return config.JSONS_FOLDER + 'trg_j_' + remove_extension(image_name) + '.json'


def image_name_to_label_path(image_name):
    return config.LABELS_FOLDER + 'trg_l_' + remove_extension(image_name) + '.tiff'


def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def create_folder_if_not_exists(folder):
    if os.path.exists(folder):
        return
    os.mkdir(folder)


def create_path(path):
    assert 'publaynet' not in path

    if path[-1] == '/':
        path = path[:-1]

    folders = path.split('/')

    cur_path = ''
    for folder in folders:
        cur_path += folder + '/'
        create_folder_if_not_exists(cur_path)

def create_weights_folder(network_config):
    create_path(network_config.WEIGHTS_FOLDER_PATH)
