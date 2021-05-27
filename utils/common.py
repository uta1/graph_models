from config import config, classifier_config
from model_processing.nets import unet


def repeat_generator(generator, args):
    while True:
        for result in generator(*args):
            yield result


def unet_for_classifier():
    return unet(
        input_size=(*config.TARGET_SIZE, 1 if config.BINARIZE else 3),
        chpt_path=classifier_config.UNET_MODEL_PATH
    )
