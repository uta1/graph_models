from lib_imports import *

from samples_maker import *
from settings import *
from trainer import *
from trg_creator import *

if __name__ == '__main__':
    if 'prepare_samples' in config.MODULES:
        prepare_samples()

    if 'prepare_trg' in config.MODULES:
        prepare_trg()

    if 'train' in config.MODULES:
        train()
