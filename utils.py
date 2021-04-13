from lib_imports import *

from geometry import *
from settings import *


def remove_extension(file_name):
    return file_name.split('.')[0]


def image_name_to_bin_path(image_name):
    return BINS_FOLDER + 'trg_b_' + remove_extension(image_name) + '.tiff'


def image_name_to_json_path(image_name):
    return JSONS_FOLDER + 'trg_j_' + remove_extension(image_name) + '.json'


def image_name_to_label_path(image_name):
    return LABELS_FOLDER + 'trg_l_' + remove_extension(image_name) + '.tiff'


def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def create_folder_if_not_exists(folder):
    if path.exists(folder):
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


def cache_and_get_indices():
    if path.exists(CACHED_LABELS_PATH) and path.exists(CACHED_ID_BY_FILE_NAME_PATH):
        with open(CACHED_LABELS_PATH, 'r') as fp:
            cached_labels = {
                int(image_id): value
                for image_id, value in json.load(fp).items()
            }
        with open(CACHED_ID_BY_FILE_NAME_PATH, 'r') as fp:
            image_id_by_file_name = json.load(fp)
    else:
        cached_labels, image_id_by_file_name = get_labels_indices()
        with open(CACHED_LABELS_PATH, 'w') as fp:
            json.dump(cached_labels, fp=fp)
        with open(CACHED_ID_BY_FILE_NAME_PATH, 'w') as fp:
            json.dump(image_id_by_file_name, fp=fp)
    return cached_labels, image_id_by_file_name


def get_labels_indices():
    labels = get_labels_full()
    labels_by_image_id = {}
    image_id_by_file_name = {}
    dims_by_image_id = {}
    for image in labels['images']:
        labels_by_image_id[image['id']] = {
            'file_name': FOLDER + image['file_name'],
            'bin_file_name': image_name_to_bin_path(image['file_name']),
            'labels_file_name': image_name_to_label_path(image['file_name']),
            'json_labels_file_name': image_name_to_json_path(image['file_name']),
            'annotations': []
        }
        dims_by_image_id[image['id']] = {
            'width': image['width'],
            'height': image['height']
        }
        image_id_by_file_name[image['file_name']] = image['id']

    for ann in labels['annotations']:
        coef_width, coef_height = get_resizing_coefs(
            dims_by_image_id[ann['image_id']]['width'],
            dims_by_image_id[ann['image_id']]['height']
        )
        labels_by_image_id[ann['image_id']]['annotations'].append(
            {
                'bbox': resize_rect(coef_width, coef_height, ann['bbox']),
                'category_id': ann['category_id']
            }
        )
    return labels_by_image_id, image_id_by_file_name


def get_labels_full():
    with open(LABELS_PATH, 'r') as fp:
        return json.load(fp)
