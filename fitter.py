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
from utils.trg_interactor import cache_and_get_images_metainfo


def _fit_network(
    images_metainfo_train,
    images_metainfo_val,
    model,
    data_generator_func,
    data_generator_args,
    network_config,
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
        steps_per_epoch=len(images_metainfo_train),
        validation_steps=len(images_metainfo_val),
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
    )


def _fit_classifier(images_metainfo_train, images_metainfo_val):
    model = classifier(input_size=(*config.ROI_EMBEDDING_SIZE,))
    unet_model = unet_for_classifier()
    lock = Lock()
    _fit_network(
        images_metainfo_train,
        images_metainfo_val,
        model,
        generate_data_classifier,
        (lock, unet_model, 'training'),
        classifier_config,
    )


def fit():
    images_metainfo_train = cache_and_get_images_metainfo('train')
    images_metainfo_val = cache_and_get_images_metainfo('val')

    if config.MODEL == 'unet':
        fit_func = _fit_unet
    if config.MODEL == 'classifier':
        fit_func = _fit_classifier
    fit_func(images_metainfo_train, images_metainfo_val)
