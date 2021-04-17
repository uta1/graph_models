from lib_imports import *

from utils.platform_based_params import folders_delim, workplace_dir


class Config:
    # Main
    MODULES = ['prepare_samples', 'prepare_trg', 'fit']
    MODEL = 'classifier'

    # Names of directories containing data
    SRC_FOLDER_NAME = 'src'  # contains train, val, test
    TRG_FOLDER_NAME = 'trg'

    TRAIN_FOLDER_NAME = 'samples'
    VAL_FOLDER_NAME = 'val'
    TEST_FOLDER_NAME = 'samples'

    # Weights saving settings
    WEIGHTS_FOLDER_NAME_TEMPLATE = '{}_weights'  # 'name_weights'
    WEIGHTS_FILE_NAME_TEMPLATE = '{epoch:03d}_epoches.chpt'

    # Logs settings
    LOGS_FOLDER_NAME = 'logs'
    LOG_FILE_NAME_TEMPLATE = 'log_{}_{}_{}.log'  # 'log_timestamp_model_lr.log'

    # Rects prediction
    MIN_OBJECT_WIDTH = 4
    MIN_OBJECT_HEIGHT = 4
    RECTS_DILATION = (7, 6)

    # Trg-creator settings
    MODES_TO_CREATE_TRG = ['train', 'val']
    TARGET_SIZE = (512, 512)  # None or tuple
    BINARIZE = True  # Bool
    BBOXES_TO_PLOT = []  # elements: 'predicted', 'labeled'

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

    def labels_file_name(self, mode):
        return self._actual_folder_name(mode) + '.json'

    def labels_file_path(self, mode):
        return self.SRC_FOLDER_PATH + self.labels_file_name(mode)

    # Directory of dataset tagged by mode

    def folder_path(self, mode):
        return self.SRC_FOLDER_PATH + self._actual_folder_name(mode) + folders_delim()

    # Directories of labels and target images

    def bins_folder_path(self, mode):
        return self.TRG_FOLDER_PATH + self._actual_folder_name(mode) + '_bins' + folders_delim()

    def labels_folder_path(self, mode):
        return self.TRG_FOLDER_PATH + self._actual_folder_name(mode) + '_labels' + folders_delim()

    def jsons_folder_path(self, mode):
        return self.TRG_FOLDER_PATH + self._actual_folder_name(mode) + '_jsons' + folders_delim()

    # Path of trg-file containing metainfo redesigned for our needs

    def cached_labels_path(self, mode):
        return self.jsons_folder_path(mode) + 'cached_' + self.labels_file_name(mode)

    # Logs properties

    @property
    def LOGS_FOLDER_PATH(self):
        return workplace_dir() + self.LOGS_FOLDER_NAME + folders_delim()

    def log_file_path_snapshot(self):
        return self.LOGS_FOLDER_PATH + self._log_file_name_snapshot()

    def _log_file_name_snapshot(self):
        return self.LOG_FILE_NAME_TEMPLATE.format(
            time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()),
            self.MODEL,
            self.LEARNING_RATE
        )

    # Private methods

    def _actual_folder_name(self, mode):
        if mode == 'train':
            return self.TRAIN_FOLDER_NAME
        if mode == 'val':
            return self.VAL_FOLDER_NAME
        if mode == 'test':
            return self.TEST_FOLDER_NAME


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
