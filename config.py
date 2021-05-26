from lib_imports import *

from utils.platform_based_params import folders_delim, workplace_dir


@dataclasses.dataclass
class Config:
    # Main
    MODULES: list = dataclasses.field(
        default_factory=lambda: ['prepare_samples', 'prepare_trg', 'fit']
    )
    MODEL: str = 'classifier'

    # Names of directories containing data
    SRC_FOLDER_NAME: str = 'src'  # contains train, val, test
    TRG_FOLDER_NAME: str = 'trg'

    TRAIN_FOLDER_NAME: str = 'samples'
    VAL_FOLDER_NAME: str = 'val'
    TEST_FOLDER_NAME: str = 'samples'

    # Weights saving settings
    WEIGHTS_FOLDER_NAME_TEMPLATE: str = '{}_weights'  # 'name_weights'
    WEIGHTS_FILE_NAME_TEMPLATE: str = '{epoch:03d}_epoches.chpt'

    # Unet samples saving settings
    UNET_SAMPLES_FOLDER_NAME: str = 'epoches_samples'
    UNET_SAMPLE_FILE_NAME_TEMPLATE: str = '{epoch:03d}_epoches.png'

    # Logs settings
    LOGS_FOLDER_NAME: str = 'logs'
    LOG_FILE_NAME_TEMPLATE: str = 'log_{}_{}.log'  # 'log_timestamp_model.log'

    # Rects prediction
    MIN_OBJECT_WIDTH: int = 4
    MIN_OBJECT_HEIGHT: int = 4
    RECTS_DILATION: tuple = (7, 6)

    # Trg-creator settings
    MODES_TO_CREATE_TRG: list = dataclasses.field(
        default_factory=lambda: ['train', 'val']
    )
    TARGET_SIZE: tuple = (512, 512)  # None or tuple
    BINARIZE: bool = True
    BBOXES_TO_PLOT: list = dataclasses.field(  # elements: 'predicted', 'labeled'
        default_factory=lambda: []
    )

    # Learning
    IMAGE_ELEM_EMBEDDING_SIZE: tuple = (16, 16, 6)


config = Config()


@dataclasses.dataclass
class NetworkConfig:
    NAME: str
    BATCH_SIZE: int


unet_config = NetworkConfig(
    NAME='unet',
    BATCH_SIZE=1,
)


classifier_config = NetworkConfig(
    NAME='classifier',
    BATCH_SIZE=16,
)
