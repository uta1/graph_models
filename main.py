from lib_imports import *
from samples_maker import *
from trainer import *
from trg_creator import *


if __name__ == '__main__':
    if 'prepare_samples' in MODULES:
        prepare_samples()

    if 'create_trg_images' in MODULES:
        create_trg_images()

    if 'train' in MODULES:
        train()
