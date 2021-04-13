from lib_imports import *
from nets import *
from settings import *
from utils import *


def train():
    cached_labels, image_id_by_file_name = cache_and_get_indices()
    for _, image_id in image_id_by_file_name.items():
        # TODO: organize traning by image_id
        pass

    model = unet(input_size=(*TARGET_SIZE, 1))
    for image in get_file_names(BINFOLDER):
        img = cv2.imread(BINFOLDER + image)
        img = np.expand_dims(img[:,:,2:], axis=0)
        print(img.shape)
        model.predict(img)
