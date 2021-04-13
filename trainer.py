from lib_imports import *

from nets import *
from settings import *
from utils import *


def generate_data(cached_labels):
    while True:
        batch_x = []
        batch_y = []
        for image_data in cached_labels.values():
            bined = cv2.imread(image_data['bin_file_name'])[:, :, 0]
            batch_x.append(np.expand_dims(bined, axis=-1))
            labels = cv2.imread(image_data['labels_file_name'])[:, :, 0]
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
    cached_labels, image_id_by_file_name = cache_and_get_indices()

    steps_per_epoch = len(cached_labels) / BATCH_SIZE + (0 if len(cached_labels) % BATCH_SIZE == 0 else 1)
    model.fit_generator(
        generate_data(cached_labels),
        steps_per_epoch=steps_per_epoch
    )
