from lib_imports import *
from consts import *
from samples_maker import *
from trg_creator import *
from utils import *
from nets import *


def process():
    model = unet(input_size=(512,512,1))
    for image in get_file_names(BINFOLDER):
        img = cv2.imread(BINFOLDER + image)
        img = np.expand_dims(img[:,:,2:], axis=0)
        print(img.shape)
        model.predict(img)


if __name__ == '__main__':
    if 'download' in MODULES:
        download()
        extract()

    if 'create_trg_images' in MODULES:
        create_trg_images()

    if 'main' in MODULES:
        process()
