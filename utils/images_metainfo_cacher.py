from lib_imports import *

from config import config
from utils.geometry import resize_rect, get_resizing_coefs
from utils.filesystem_helper import (
    cached_labels_path,
    image_name_to_path,
    image_name_to_bin_path,
    image_name_to_label_path,
    labels_file_path,
)


def cache_and_get_images_metainfo(mode):
    if os.path.exists(cached_labels_path(mode)):
        with open(cached_labels_path(mode), 'r') as fp:
            images_metainfo = {
                int(image_id): value
                for image_id, value in json.load(fp).items()
            }
    else:
        images_metainfo = _get_images_metainfo(mode)
        with open(cached_labels_path(mode), 'w') as fp:
            json.dump(images_metainfo, fp=fp)
    return images_metainfo


def _get_images_metainfo(mode):
    labels = _get_images_fullinfo(mode)
    images_metainfo = {}
    dims_by_image_id = {}
    for image in labels['images']:
        images_metainfo[image['id']] = {
            'file_path': image_name_to_path(mode, image['file_name']),
            'bin_file_path': image_name_to_bin_path(mode, image['file_name']),
            'label_file_path': image_name_to_label_path(mode, image['file_name']),
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


def _get_images_fullinfo(mode):
    with open(labels_file_path(mode), 'r') as fp:
        return json.load(fp)
