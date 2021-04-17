from lib_imports import *


def _reshaped_image_from_path(path, axis):
    image = cv2.imread(path)[:, :, 0]
    return np.expand_dims(image, axis=axis)


def np_image_from_path(path):
    return _reshaped_image_from_path(path, -1)


def np_monobatch_from_path(path):
    return _reshaped_image_from_path(path, [0, 3])
