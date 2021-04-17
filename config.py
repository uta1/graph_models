from lib_imports import *

from utils.filesystem_helper import create_path
from utils.platform_based_params import folders_delim
from utils.platform_based_params import workplace_dir


class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg', 'fit']
    MODEL = 'classifier'
    MODE = 'samples'

    # Names of directories containing data
    SRC_FOLDER_NAME = 'src'
    TRG_FOLDER_NAME = 'trg'

    # Weights saving settings
    WEIGHTS_FOLDER_NAME_TEMPLATE = '{}_weights'  # 'name_weights'
    WEIGHTS_FILE_NAME_TEMPLATE = '{epoch:03d}_epoches.chpt'

    # Logs settings
    LOGS_FOLDER_NAME = 'logs'
    LOG_FILE_NAME_TEMPLATE = 'log_{}_{}_{}_{}.log'  # 'log_mode_model_lr_timestamp.log'

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
    IMAGE_ELEM_EMBEDDING_SIZE = (16, 16, 6)
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
            self.MODEL,
            self.LEARNING_RATE,
            time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        )

    def is_mode_trainable(self):
        return self.MODE in ['samples', 'train']


config = Config()


@dataclasses.dataclass
class NetworkConfig:
    NAME: str
    BATCH_SIZE: int

    @property
    def WEIGHTS_FOLDER_PATH(self):
        return workplace_dir() + config.WEIGHTS_FOLDER_NAME_TEMPLATE.format(self.NAME) + folders_delim()

    @property
    def WEIGHTS_FILE_PATH_TEMPLATE(self):
        return self.WEIGHTS_FOLDER_PATH + config.WEIGHTS_FILE_NAME_TEMPLATE


unet_config = NetworkConfig(
    NAME='unet',
    BATCH_SIZE=1,
)


classifier_config = NetworkConfig(
    NAME='classifier',
    BATCH_SIZE=16,
)
