from lib_imports import *


def _reshaped_image_from_path(path, axis, drop_last_dim):
    image = cv2.imread(path)
    if not drop_last_dim:
        return image
    return np.expand_dims(image[:, :, 0], axis=axis)


def np_image_from_path(path, drop_last_dim=True):
    return _reshaped_image_from_path(path, -1, drop_last_dim=drop_last_dim)


def np_monobatch_from_path(path, drop_last_dim=True):
    return _reshaped_image_from_path(path, [0, 3], drop_last_dim=drop_last_dim)
