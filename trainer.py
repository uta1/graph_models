from lib_imports import *

from nets import *
from config import *
from utils.filesystem_helper import create_path
from utils.images_metainfo_cacher import cache_and_get_images_metainfo


def generate_data(images_metainfo):
    while True:
        batch_x = []
        batch_y = []
        for image_metainfo in images_metainfo.values():
            bined = cv2.imread(image_metainfo['bin_file_path'])[:, :, 0]
            batch_x.append(np.expand_dims(bined, axis=-1))
            labels = cv2.imread(image_metainfo['label_file_path'])[:, :, 0]
            batch_y.append(np.expand_dims(labels[:16,:16], axis=-1))

            if len(batch_x) == config.BATCH_SIZE:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

        if len(batch_x) > 0:
            yield np.array(batch_x), np.array(batch_y)
    return


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


def train():
    model = unet(input_size=(*config.TARGET_SIZE, 1))
    images_metainfo = cache_and_get_images_metainfo()
    create_path(config.WEIGHTS_FOLDER_PATH)

    steps_per_epoch = len(images_metainfo) / config.BATCH_SIZE + \
                      (0 if len(images_metainfo) % config.BATCH_SIZE == 0 else 1)
    model.fit_generator(
        generate_data(images_metainfo),
        steps_per_epoch=steps_per_epoch,
        epochs=3000,
        callbacks=[
            LoggerCallback(),
            ModelCheckpoint(
                filepath=config.WEIGHTS_FILE_PATH_TEMPLATE,
                save_weights_only=False
            )
        ],
    )
