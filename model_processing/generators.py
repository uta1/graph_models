from lib_imports import *

from config import config, unet_config, classifier_config
from utils.cv2_utils import np_image_from_path, np_monobatch_from_path


x_to_eval = None

def generate_data_unet(images_metainfo):
    global x_to_eval

    batch_x = []
    batch_y = []
    for image_metainfo in images_metainfo.values():
        batch_x.append(
            np_image_from_path(
                image_metainfo['bin_file_path'],
                binarized=config.BINARIZE
            )
        )
        batch_y.append(
            np_image_from_path(
                image_metainfo['label_file_path'],
                binarized=True
            )
        )

        if len(batch_x) == unet_config.BATCH_SIZE:
            if x_to_eval is None:
                x_to_eval = np.array(batch_x)
            yield np.array(batch_x), np.array(batch_y)
            batch_x = []
            batch_y = []

    if len(batch_x) > 0:
        yield np.array(batch_x), np.array(batch_y)


def _resize_suboutput(unet_output, x, y, w, h):
    return cv2.resize(
        unet_output[0, y:y+h, x:x+w, :],
        (config.ROI_EMBEDDING_SIZE[1], config.ROI_EMBEDDING_SIZE[0]),
        interpolation=cv2.INTER_NEAREST
    )


def generate_data_classifier(
    images_metainfo,
    lock,
    unet_model,
    strategy
):
    batch_x = []
    batch_y = []
    for image_metainfo in images_metainfo.values():
        unet_input = np_monobatch_from_path(
            image_metainfo['bin_file_path'],
            binarized=config.BINARIZE
        )
        lock.acquire()
        unet_output = unet_model.predict(unet_input)
        lock.release()

        if strategy == 'training':
            for ann in image_metainfo['annotations']:
                x, y, w, h = ann['bbox']

                batch_x.append(_resize_suboutput(unet_output, x, y, w, h))
                batch_y.append(ann['category_id'])

                if len(batch_x) == classifier_config.BATCH_SIZE:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []

        if strategy == 'prediction':
            for x, y, w, h in image_metainfo['rois']:
                batch_x.append(_resize_suboutput(unet_output, x, y, w, h))
            yield np.array(batch_x), image_metainfo['annotations'], image_metainfo['rois']
            batch_x = []

    if strategy == 'training' and len(batch_x) > 0:
        yield np.array(batch_x), np.array(batch_y)
