from lib_imports import *

from config import config, classifier_config
from logger import logger
from model_processing.nets import unet


def repeat_generator(generator, args):
    while True:
        for result in generator(*args):
            yield result


def unet_for_classifier():
    model = unet(
        input_size=(*config.TARGET_SIZE, 1 if config.BINARIZE else 3),
        chpt_path=classifier_config.UNET_MODEL_PATH
    )
    model = Model(model.inputs, model.layers[-2].output)

    model.summary(print_fn=lambda x: logger.log(x))

    return model
