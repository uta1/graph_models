from lib_imports import *
import argparse
import tensorflow as tf
from logger import logger
from fitter import generate_data_unet
from utils.images_metainfo_cacher import cache_and_get_images_metainfo

parser = argparse.ArgumentParser()
parser.add_argument(
    "dir",
    help="what dir load model from",
    type=str
)
args = parser.parse_args()

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

        metrics_estimator = tf.keras.metrics.MeanIoU(num_classes=6)
        counter = 0
        for x, y_true in generate_data_unet(images_metainfo_val):
            y_pred = np.argmax(model.predict(x), axis=-1)
            metrics_estimator.update_state(y_pred, y_true)
            counter += 1
            print(epoch_name, str(counter) + '/' + str(len(images_metainfo_val.values())))
        logger.log(epoch_name + ': ' + str(metrics_estimator.result().numpy()))

