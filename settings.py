from lib_imports import *


@dataclasses.dataclass
class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg']
    MODE = 'samples'

    LABELS = MODE + '.json'

    # Directories of data
    SRC_FOLDER = '../src/'
    TRG_FOLDER = '../trg/'

    FOLDER = SRC_FOLDER + MODE + '/'

    # Directories of labels and target images
    BINS_FOLDER = TRG_FOLDER + MODE + '_bins/'
    LABELS_FOLDER = TRG_FOLDER + MODE + '_labels/'
    JSONS_FOLDER = TRG_FOLDER + MODE + '_jsons/'

    # Labels creation
    LABELS_PATH = SRC_FOLDER + LABELS
    CACHED_LABELS_PATH = JSONS_FOLDER + 'cached_' + LABELS

    # Rects prediction
    MIN_OBJECT_WIDTH = 4
    MIN_OBJECT_HEIGHT = 4
    RECTS_DILATION = (7, 6)

    # Trg-creator settings
    TARGET_SIZE = (1024, 512)  # None or tuple
    BINARIZE = True  # Bool
    BBOXES_TO_PLOT = []  # elements: 'predicted', 'labeled'
    SAVE_JSONS = False

    # Learning
    BATCH_SIZE = 1


config = Config()
