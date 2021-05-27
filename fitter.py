from lib_imports import *

from config import config, unet_config, classifier_config
from model_processing.nets import unet, classifier
from model_processing.callbacks import LoggerCallback, SampleSaverCallback
from model_processing.generators import generate_data_unet, generate_data_classifier
from utils.filesystem_helper import (
    create_path,
    create_weights_folder,
    weights_file_path_template,
    create_unet_samples_folder
)
from utils.common import repeat_generator, unet_for_classifier
from utils.cv2_utils import np_image_from_path, np_monobatch_from_path
from utils.images_metainfo_cacher import cache_and_get_images_metainfo


def _steps_per_epoch(items, batch_size):
    return items / batch_size + \
           (0 if items % batch_size == 0 else 1)


def _len_annotations(images_metainfo):
    res = 0
    for image_metainfo in images_metainfo.values():
        res += len(image_metainfo['annotations'])
    return res


def _fit_network(
    images_metainfo_train,
    images_metainfo_val,
    model,
    data_generator_func,
    data_generator_args,
    network_config,
    data_len_func
):
    create_weights_folder(network_config)

    callbacks = [
        LoggerCallback(),
        ModelCheckpoint(
            filepath=weights_file_path_template(network_config),
            save_weights_only=False
        )
    ]
    if config.MODEL == 'unet':
        create_unet_samples_folder()
        callbacks.append(SampleSaverCallback())

    model.fit_generator(
        repeat_generator(data_generator_func, (images_metainfo_train, *data_generator_args)),
        validation_data=repeat_generator(data_generator_func, (images_metainfo_val, *data_generator_args)),
        steps_per_epoch=_steps_per_epoch(data_len_func(images_metainfo_train), network_config.BATCH_SIZE),
        validation_steps=_steps_per_epoch(data_len_func(images_metainfo_val), network_config.BATCH_SIZE),
        epochs=3000,
        callbacks=callbacks,
    )


def _fit_unet(images_metainfo_train, images_metainfo_val):
    _fit_network(
        images_metainfo_train,
        images_metainfo_val,
        unet(input_size=(*config.TARGET_SIZE, 1 if config.BINARIZE else 3)),
        generate_data_unet,
        (),
        unet_config,
        len,
    )


def _fit_classifier(images_metainfo_train, images_metainfo_val):
    model = classifier(input_size=(*config.IMAGE_ELEM_EMBEDDING_SIZE, ))
    unet_model = unet_for_classifier()
    lock = Lock()
    _fit_network(
        images_metainfo_train,
        images_metainfo_val,
        model,
        generate_data_classifier,
        (lock, unet_model, 'by_config'),
        classifier_config,
        _len_annotations,
    )


def fit():
    images_metainfo_train = cache_and_get_images_metainfo('train')
    images_metainfo_val = cache_and_get_images_metainfo('val')

    if config.MODEL == 'unet':
        fit_func = _fit_unet
    if config.MODEL == 'classifier':
        fit_func = _fit_classifier
    fit_func(images_metainfo_train, images_metainfo_val)
