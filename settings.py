from lib_imports import *


@dataclasses.dataclass
class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg', 'train']
    MODE = 'samples'

    # Directories of data
    SRC_FOLDER = '../src/'
    TRG_FOLDER = '../trg/'

    # Rects prediction
    MIN_OBJECT_WIDTH = 4
    MIN_OBJECT_HEIGHT = 4
    RECTS_DILATION = (7, 6)

    # Trg-creator settings
    TARGET_SIZE = (512, 512)  # None or tuple
    BINARIZE = True  # Bool
    BBOXES_TO_PLOT = []  # elements: 'predicted', 'labeled'
    SAVE_JSONS = False

    # Learning
    BATCH_SIZE = 1

    # Name of src-file containing metainfo
    # Available only for samples, train and val

    @property
    def LABELS(self):
        return self.MODE + '.json'

    @property
    def LABELS_PATH(self):
        return self.SRC_FOLDER + self.LABELS

    # Directory of dataset tagged by mode

    @property
    def FOLDER(self):
        return self.SRC_FOLDER + self.MODE + '/'

    # Directories of labels and target images

    @property
    def BINS_FOLDER(self):
        return self.TRG_FOLDER + self.MODE + '_bins/'

    @property
    def LABELS_FOLDER(self):
        return self.TRG_FOLDER + self.MODE + '_labels/'

    @property
    def JSONS_FOLDER(self):
        return self.TRG_FOLDER + self.MODE + '_jsons/'

    # Path of trg-file containing metainfo redesigned for our needs

    @property
    def CACHED_LABELS_PATH(self):
        return self.JSONS_FOLDER + 'cached_' + self.LABELS


config = Config()
