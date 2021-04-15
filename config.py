from lib_imports import *

from utils.platform_based_params import folders_delim
from utils.platform_based_params import workplace_dir


class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg', 'train']
    MODE = 'samples'

    # Names of directories containing data
    SRC_FOLDER_NAME = 'src'
    TRG_FOLDER_NAME = 'trg'

    # Logs settings
    WEIGHTS_FOLDER_NAME = 'weights'
    LOGS_FOLDER_NAME = 'logs'
    LOG_FILE_NAME_TEMPLATE = 'log_{}_{}_{}.log'  # 'log_mode_lr_timestamp.log'

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
        return workplace_dir() + self.SRC_FOLDER_NAME + folders_delim()

    @property
    def TRG_FOLDER_PATH(self):
        return workplace_dir() + self.TRG_FOLDER_NAME + folders_delim()

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
        return self.SRC_FOLDER_PATH + self.MODE + folders_delim()

    # Directories of labels and target images

    @property
    def BINS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_bins' + folders_delim()

    @property
    def LABELS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_labels' + folders_delim()

    @property
    def JSONS_FOLDER(self):
        return self.TRG_FOLDER_PATH + self.MODE + '_jsons' + folders_delim()

    # Path of trg-file containing metainfo redesigned for our needs

    @property
    def CACHED_LABELS_PATH(self):
        return self.JSONS_FOLDER + 'cached_' + self.LABELS

    # Logs properties

    @property
    def LOGS_FOLDER_PATH(self):
        return workplace_dir() + self.LOGS_FOLDER_NAME + folders_delim()

    def log_file_path_snapshot(self):
        return self.LOGS_FOLDER_PATH + self.log_file_name_snapshot()

    def log_file_name_snapshot(self):
        return self.LOG_FILE_NAME_TEMPLATE.format(
            self.MODE,
            self.LEARNING_RATE,
            time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        )


config = Config()
