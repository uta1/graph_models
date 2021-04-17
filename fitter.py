from lib_imports import *

from config import config, unet_config, classifier_config
from nets import unet, classifier
from logger import logger
from utils.filesystem_helper import create_path, create_weights_folder
from utils.cv2_utils import np_image_from_path, np_monobatch_from_path
from utils.images_metainfo_cacher import cache_and_get_images_metainfo


def generate_data_unet(images_metainfo):
    while True:
        batch_x = []
        batch_y = []
        for image_metainfo in images_metainfo.values():
            batch_x.append(np_image_from_path(image_metainfo['bin_file_path']))
            batch_y.append(np_image_from_path(image_metainfo['label_file_path']))

            if len(batch_x) == unet_config.BATCH_SIZE:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

        if len(batch_x) > 0:
            yield np.array(batch_x), np.array(batch_y)
    return


def generate_data_classifier(images_metainfo, unet_model, graph):
    while True:
        batch_x = []
        batch_y = []
        for image_metainfo in images_metainfo.values():
            unet_input = np_monobatch_from_path(image_metainfo['bin_file_path'])
            with graph.as_default():
                unet_output = unet_model.predict(unet_input)

            for ann in image_metainfo['annotations']:
                x, y, w, h = ann['bbox']
                batch_x.append(
                    cv2.resize(
                        unet_output[0, y:y+h, x:x+w, :],
                        (config.IMAGE_ELEM_EMBEDDING_SIZE[1], config.IMAGE_ELEM_EMBEDDING_SIZE[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                )
                batch_y.append(ann['category_id'])

                if len(batch_x) == classifier_config.BATCH_SIZE:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []

        if len(batch_x) > 0:
            yield np.array(batch_x), np.array(batch_y)


class LoggerCallback(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        logger.log('Epoch {} began'.format(epoch), print_timestamp=True)

    def on_batch_end(self, batch, logs):
        logger.log(
            'batch: {} train_loss: {} train_categorical_accuracy: {} train_iou_score: {}'.format(
                batch,
                logs.get('loss'),
                logs.get('sparse_categorical_accuracy'),
                logs.get('iou_score')
            )
        )

    def on_epoch_end(self, epoch, logs={}):
        logger.log(
            'epoch_loss: {} epoch_categorical_accuracy: {} epoch_iou_score: {}'.format(
                logs.get('loss'),
                logs.get('sparse_categorical_accuracy'),
                logs.get('iou_score')
            )
        )
        logger.log(
            'epoch_val_loss: {} epoch_val_categorical_accuracy: {} epoch_val_iou_score: {}'.format(
                logs.get('val_loss'),
                logs.get('val_sparse_categorical_accuracy'),
                logs.get('val_iou_score')
            )
        )


def _steps_per_epoch(items, batch_size):
    return items / batch_size + \
           (0 if items % batch_size == 0 else 1)


def _len_annotations(images_metainfo):
    res = 0
    for image_metainfo in images_metainfo.values():
        res += len(image_metainfo['annotations'])
    return res


def _fit_network(
    images_metainfo,
    model,
    data_generator_func,
    data_generator_args,
    network_config,
    data_len_func
):
    create_weights_folder(network_config)

    model.fit_generator(
        data_generator_func(images_metainfo, *data_generator_args),
        steps_per_epoch=_steps_per_epoch(data_len_func(images_metainfo), network_config.BATCH_SIZE),
        epochs=3000,
        callbacks=[
            LoggerCallback(),
            ModelCheckpoint(
                filepath=network_config.WEIGHTS_FILE_PATH_TEMPLATE,
                save_weights_only=False
            )
        ],
    )


def _fit_unet(images_metainfo):
    _fit_network(
        images_metainfo,
        unet(input_size=(*config.TARGET_SIZE, 1)),
        generate_data_unet,
        (),
        unet_config,
        len,
    )


def _fit_classifier(images_metainfo):
    model = classifier(input_size=(*config.IMAGE_ELEM_EMBEDDING_SIZE, ))
    graph = tf.get_default_graph()
    unet_model = unet(input_size=(*config.TARGET_SIZE, 1))
    _fit_network(
        images_metainfo,
        model,
        generate_data_classifier,
        (unet_model, graph,),
        classifier_config,
        _len_annotations,
    )


def fit():
    images_metainfo = cache_and_get_images_metainfo()
    if config.MODEL == 'unet':
        _fit_unet(images_metainfo)
    if config.MODEL == 'classifier':
        _fit_classifier(images_metainfo)