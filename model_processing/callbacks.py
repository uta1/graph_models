from lib_imports import *

from logger import logger
from model_processing import generators
from utils.filesystem_helper import unet_sample_file_path_template


class LoggerCallback(Callback):
    losses = []
    sparse_categorical_accuracies = []
    iou_scores = []

    val_losses = []
    val_sparse_categorical_accuracies = []
    val_iou_scores = []

    def on_epoch_begin(self, epoch, logs={}):
        logger.log('Epoch {} began'.format(epoch), print_timestamp=True)

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.sparse_categorical_accuracies.append(logs.get('sparse_categorical_accuracy'))

        self.val_losses.append(logs.get('val_loss'))
        self.val_sparse_categorical_accuracies.append(logs.get('val_sparse_categorical_accuracy'))

        logger.log(
            'epochs_losses: {}'.format(
                str(self.losses)
            )
        )
        logger.log(
            'epochs_sparse_categorical_accuracies: {}'.format(
                str(self.sparse_categorical_accuracies)
            )
        )

        logger.log(
            'epochs_val_losses: {}'.format(
                str(self.val_losses)
            )
        )
        logger.log(
            'epochs_val_sparse_categorical_accuracies: {}'.format(
                str(self.val_sparse_categorical_accuracies)
            )
        )


class SampleSaverCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        predicted_sample = self.model.predict(generators.x_to_eval)[0,:,:,:]
        to_save = np.argmax(predicted_sample, axis=-1)

        cv2.imwrite(
            unet_sample_file_path_template().format(epoch=epoch),
            to_save * 50
        )
