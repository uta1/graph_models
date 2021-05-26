from lib_imports import *

from samples_maker import prepare_samples
from config import config
from fitter import fit
from trg_creator import prepare_trg


if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    np.set_printoptions(threshold=sys.maxsize)

    if 'prepare_samples' in config.MODULES:
        prepare_samples()

    if 'prepare_trg' in config.MODULES:
        prepare_trg()

    if 'fit' in config.MODULES:
        fit()
