from lib_imports import *

from geometry import *
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


def cache_and_get_images_metainfo():
    if os.path.exists(config.CACHED_LABELS_PATH):
        with open(config.CACHED_LABELS_PATH, 'r') as fp:
            images_metainfo = {
                int(image_id): value
                for image_id, value in json.load(fp).items()
            }
    else:
        images_metainfo = get_images_metainfo()
        with open(config.CACHED_LABELS_PATH, 'w') as fp:
            json.dump(images_metainfo, fp=fp)
    return images_metainfo


def get_images_metainfo():
    labels = get_images_fullinfo()
    images_metainfo = {}
    dims_by_image_id = {}
    for image in labels['images']:
        images_metainfo[image['id']] = {
            'file_path': image_name_to_path(image['file_name']),
            'bin_file_path': image_name_to_bin_path(image['file_name']),
            'label_file_path': image_name_to_label_path(image['file_name']),
            'json_label_file_path': image_name_to_json_path(image['file_name']),
            'annotations': []
        }
        dims_by_image_id[image['id']] = {
            'width': image['width'],
            'height': image['height']
        }

    for ann in labels['annotations']:
        image_id = ann['image_id']
        coef_width, coef_height = get_resizing_coefs(
            dims_by_image_id[image_id]['width'],
            dims_by_image_id[image_id]['height']
        )
        images_metainfo[image_id]['annotations'].append(
            {
                'bbox': resize_rect(coef_width, coef_height, ann['bbox']),
                'category_id': ann['category_id']
            }
        )
    return images_metainfo


def get_images_fullinfo():
    with open(config.LABELS_PATH, 'r') as fp:
        return json.load(fp)
