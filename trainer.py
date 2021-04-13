from lib_imports import *

from nets import *
from settings import *
from utils import *


def train():
    model = unet(input_size=(*TARGET_SIZE, 1))

    cached_labels, image_id_by_file_name = cache_and_get_indices()
    for _, image_id in image_id_by_file_name.items():
        bined = cv2.imread(cached_labels[image_id]['bin_file_name'])[:, :, 0]
        bined = np.expand_dims(bined, axis=[0, 3])
        labels = cv2.imread(cached_labels[image_id]['labels_file_name'])[:, :, 0]
        labels = np.expand_dims(labels, axis=[0, 3])
        model.fit(bined, labels)
