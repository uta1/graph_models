from lib_imports import *

from config import config
from utils.geometry import resize_rect, get_resizing_coefs
from utils.filesystem_helper import (
    image_name_to_path,
    image_name_to_bin_path,
    image_name_to_json_path,
    image_name_to_label_path,
)


def cache_and_get_images_metainfo():
    if os.path.exists(config.CACHED_LABELS_PATH):
        with open(config.CACHED_LABELS_PATH, 'r') as fp:
            images_metainfo = {
                int(image_id): value
                for image_id, value in json.load(fp).items()
            }
    else:
        images_metainfo = _get_images_metainfo()
        with open(config.CACHED_LABELS_PATH, 'w') as fp:
            json.dump(images_metainfo, fp=fp)
    return images_metainfo


def _get_images_metainfo():
    labels = _get_images_fullinfo()
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


def _get_images_fullinfo():
    with open(config.LABELS_PATH, 'r') as fp:
        return json.load(fp)
