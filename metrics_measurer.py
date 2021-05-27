from lib_imports import *
import argparse
import tensorflow as tf
from logger import logger
from model_processing.generators import generate_data_unet, generate_data_classifier
from model_processing.metrics import ObjectDetectionRate
from utils.common import unet_for_classifier
from utils.images_metainfo_cacher import cache_and_get_images_metainfo


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir',
    help='what dir load model from',
    type=str
)
parser.add_argument(
    '--metrics',
    help='name of metrics to calculate',
    type=str
)


def _process_epoch_internal(model, metrics_estimator, generator):
    counter = 0
    for generated_data in generator:
        y_pred = np.argmax(model.predict(generated_data[0]), axis=-1)
        metrics_estimator.update_state(y_pred, *(generated_data[1:]))
        counter += 1
        print(epoch_name, str(counter) + '/' + str(len(images_metainfo_val.values())))
    logger.log(epoch_name + ': ' + str(metrics_estimator.result().numpy()))


def _process_epoch_mean_iou(model, images_metainfo_val):
    metrics_estimator = tf.keras.metrics.MeanIoU(num_classes=6)
    generator = generate_data_unet(images_metainfo_val)

    _process_epoch_internal(model, metrics_estimator, generator)


def _process_epoch_object_detection_rate(model, images_metainfo_val):
    unet_model = unet_for_classifier()
    lock = Lock()

    metrics_estimator = ObjectDetectionRate()
    generator = generate_data_classifier(images_metainfo_val, lock, unet_model, 'by_image')

    _process_epoch_internal(model, metrics_estimator, generator)


def _process_epoch(model, images_metainfo_val, args):
    if args.metrics == 'mean_iou':
        _process_epoch_mean_iou(model, images_metainfo_val)
    if args.metrics == 'object_detection_rate':
        _process_epoch_object_detection_rate(model, images_metainfo_val)


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    np.set_printoptions(threshold=sys.maxsize)

    images_metainfo_val = cache_and_get_images_metainfo('val')

    args = parser.parse_args()
    dir_path = args.dir
    if dir_path[-1] != '/':
        dir_path[-1] += '/'
    for epoch_name in sorted(os.listdir(args.dir)):
        model = load_model(dir_path + epoch_name)
        _process_epoch(model, images_metainfo_val, args)
