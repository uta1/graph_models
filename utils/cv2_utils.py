from lib_imports import *

def np_image_from_path(path, binarized):
    image = cv2.imread(path)
    return image[:, :, :1] if binarized else image


def np_monobatch_from_path(path, binarized):
    return np.expand_dims(np_image_from_path(path, binarized), axis=0)
