from lib_imports import *

from nets import *
from settings import *
from utils import *


def generate_data(images_metainfo):
    while True:
        batch_x = []
        batch_y = []
        for image_metainfo in images_metainfo.values():
            bined = cv2.imread(image_metainfo['bin_file_path'])[:, :, 0]
            batch_x.append(np.expand_dims(bined, axis=-1))
            labels = cv2.imread(image_metainfo['label_file_path'])[:, :, 0]
            batch_y.append(np.expand_dims(labels, axis=-1))

            if len(batch_x) == BATCH_SIZE:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

        if len(batch_x) > 0:
            yield np.array(batch_x), np.array(batch_y)
    return


def train():
    model = unet(input_size=(*TARGET_SIZE, 1))
    images_metainfo = cache_and_get_images_metainfo()

    steps_per_epoch = len(images_metainfo) / BATCH_SIZE + (0 if len(images_metainfo) % BATCH_SIZE == 0 else 1)
    model.fit_generator(
        generate_data(images_metainfo),
        steps_per_epoch=steps_per_epoch
    )
