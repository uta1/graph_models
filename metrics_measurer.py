from lib_imports import *
import argparse
import tensorflow as tf

from logger import logger
from model_processing.generators import generate_data_unet, generate_data_classifier
from model_processing.metrics import ObjectDetectionRate
from utils.common import unet_for_classifier
from utils.trg_interactor import cache_and_get_images_metainfo


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    help='what dir load model from',
    type=str,
    required=True
)
parser.add_argument(
    '--metrics',
    help='name of metrics to calculate',
    type=str,
    required=True
)
parser.add_argument(
    '--iou_threshold',
    help='IoU threshold to make a verdict of object detection',
    type=float,
    default=0.5
)


def _process_epoch_internal(epoch_name, model, metrics_estimator, generator):
    counter = 0
    for generated_data in generator:
        x, estimator_args = generated_data[0], generated_data[1:]
        y_pred = np.argmax(model.predict(x), axis=-1)
        metrics_estimator.update_state(y_pred, *estimator_args)
        counter += 1
        print(epoch_name, str(counter) + '/' + str(len(images_metainfo_val.values())))
    logger.log(epoch_name + ': ' + str(metrics_estimator.result().numpy()))


def _process_epoch_mean_iou(epoch_name, model, images_metainfo_val, *args):
    metrics_estimator = tf.keras.metrics.MeanIoU(num_classes=6)
    generator = generate_data_unet(images_metainfo_val)

    _process_epoch_internal(epoch_name, model, metrics_estimator, generator)


def _process_epoch_object_detection_rate(epoch_name, model, images_metainfo_val, process_args, user_args):
    metrics_estimator = ObjectDetectionRate(iou_threshold=user_args.iou_threshold)
    generator = generate_data_classifier(images_metainfo_val, *process_args, 'measure_metrics')

    _process_epoch_internal(epoch_name, model, metrics_estimator, generator)


def _process_epoches(images_metainfo_val, process_epoch_impl, process_args, user_args):
    for epoch_name in sorted(os.listdir(user_args.dir)):
        model = load_model(dir_path + epoch_name)
        process_epoch_impl(epoch_name, model, images_metainfo_val, process_args, user_args)


def process_params(args):
    if args.metrics == 'mean_iou':
        return _process_epoch_mean_iou, ()
    if args.metrics == 'object_detection_rate':
        lock = Lock()
        unet_model = unet_for_classifier()
        return _process_epoch_object_detection_rate, (lock, unet_model,)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    np.set_printoptions(threshold=sys.maxsize)

    images_metainfo_val = cache_and_get_images_metainfo('val')

    user_args = parser.parse_args()
    logger.log('metrics-measurer args: ' + str(user_args))
    dir_path = user_args.dir
    if dir_path[-1] != '/':
        dir_path[-1] += '/'
    process_epoch_impl, process_args = process_params(user_args)
    _process_epoches(images_metainfo_val, process_epoch_impl, process_args, user_args)
