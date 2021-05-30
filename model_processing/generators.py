from lib_imports import *

from config import config, unet_config, classifier_config
from utils.common import unet_for_classifier
from utils.cv2_utils import np_image_from_path, np_monobatch_from_path
from utils.geometry import rects_intersection


x_to_eval = None

# batch_size is hardcoded to 1
def generate_data_unet(images_metainfo):
    global x_to_eval

    for image_metainfo in images_metainfo.values():
        batch_x = [
            np_image_from_path(
                image_metainfo['bin_file_path'],
                binarized=config.BINARIZE
            )
        ]
        batch_y = [
            np_image_from_path(
                image_metainfo['label_file_path'],
                binarized=True
            )
        ]

        if x_to_eval is None:
            x_to_eval = np.array(batch_x)
        yield np.array(batch_x), np.array(batch_y)


def _resize_suboutput(unet_output, roi):
    x, y, w, h = roi
    return cv2.resize(
        unet_output[0, y:y+h, x:x+w, :],
        (config.ROI_EMBEDDING_SIZE[1], config.ROI_EMBEDDING_SIZE[0]),
        interpolation=cv2.INTER_NEAREST
    )


def _calc_category_id(roi, anns):
    for ann in anns:
        intersection_square = rects_intersection(roi, ann['bbox'])
        if intersection_square > classifier_config.IOU_DETECTION_THRESHOLD * (roi[-1] * roi[-2]):
            return ann['category_id']

    return 0


def generate_data_classifier(
    images_metainfo,
    lock,
    unet_model,
    strategy
):
    for image_metainfo in images_metainfo.values():
        unet_input = np_monobatch_from_path(
            image_metainfo['bin_file_path'],
            binarized=config.BINARIZE
        )
        lock.acquire()
        unet_output = unet_model.predict(unet_input)
        lock.release()

        batch_x = [
            _resize_suboutput(unet_output, roi)
            for roi in image_metainfo['rois']
        ]

        if strategy == 'measure_metrics':
            yield np.array(batch_x), image_metainfo['annotations'], image_metainfo['rois']
        if strategy == 'training':
            batch_y = [
                _calc_category_id(roi, image_metainfo['annotations'])
                for roi in image_metainfo['rois']
            ]
            yield np.array(batch_x), np.array(batch_y)
