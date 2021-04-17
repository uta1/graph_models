from lib_imports import *

from samples_maker import *
from config import *
from fitter import *
from trg_creator import *

if __name__ == '__main__':
    if 'prepare_samples' in config.MODULES:
        prepare_samples()

    if 'prepare_trg' in config.MODULES:
        prepare_trg()

    if 'fit_unet' in config.MODULES:
        fit_unet()

    if 'fit_classifier' in config.MODULES:
        fit_classifier()
