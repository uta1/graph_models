from lib_imports import *


class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg', 'train']
    MODE = 'samples'

    # Names of directories containing data
    SRC_FOLDER_NAME = 'src'
    TRG_FOLDER_NAME = 'trg'

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
    LEARNING_RATE = 1e-4

    # Paths of data

    @property
    def SRC_FOLDER_PATH(self):
        return self._workplace_dir + self.SRC_FOLDER_NAME + self._folders_delim

    @property
    def TRG_FOLDER_PATH(self):
        return self._workplace_dir + self.TRG_FOLDER_NAME + self._folders_delim

    # Name of src-file containing metainfo
    # Available only for samples, train and val

    @property
    def LABELS(self):
        return self.MODE + '.json'

    @property
    def LABELS_PATH(self):
        return self.SRC_FOLDER_PATH + self.LABELS

    # Directory of dataset tagged by mode

    @property
    def FOLDER(self):
        return self.SRC_FOLDER_PATH + self.MODE + self._folders_delim

    # Directories of labels and target images

    @property
    def BINS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_bins' + self._folders_delim

    @property
    def LABELS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_labels' + self._folders_delim

    @property
    def JSONS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_jsons' + self._folders_delim

    # Path of trg-file containing metainfo redesigned for our needs

    @property
    def CACHED_LABELS_PATH(self):
        return self.JSONS_FOLDER + 'cached_' + self.LABELS

    @property
    def _folders_delim(self):
        if platform.system() == 'Linux':
            return '/'
        return '\\'

    @property
    def _workplace_dir(self):
        if platform.system() == 'Linux':
            return '../'
        raise


config = Config()
