from lib_imports import *

from samples_maker import prepare_samples
from config import config
from fitter import fit
from trg_creator import prepare_trg

if __name__ == '__main__':
    if 'prepare_samples' in config.MODULES:
        prepare_samples()

    if 'prepare_trg' in config.MODULES:
        prepare_trg()

    if 'fit' in config.MODULES:
        fit()
