import typing as tp

from lib_imports import *

from utils.platform_based_params import folders_delim, workplace_dir


@dataclasses.dataclass
class Config:
    # Main
    MODULES: list = dataclasses.field(
        default_factory=lambda: ['prepare_trg', 'fit']
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
    BINARIZE: bool = False
    BBOXES_TO_PLOT: list = dataclasses.field(  # elements: 'predicted', 'labeled'
        default_factory=lambda: []
    )

    # Learning
    ROI_EMBEDDING_SIZE: tuple = (16, 16, 64)


config = Config()


@dataclasses.dataclass
class NetworkConfigBase:
    NAME: str


@dataclasses.dataclass
class UnetConfig(NetworkConfigBase):
    pass


@dataclasses.dataclass
class ClassifierConfig(NetworkConfigBase):
    UNET_MODEL_PATH: tp.Optional[str]
    IOU_DETECTION_THRESHOLD: float


unet_config = UnetConfig(
    NAME='unet',
)


classifier_config = ClassifierConfig(
    NAME='classifier',
    UNET_MODEL_PATH='../unet_weights/001_epoches.chpt',
    IOU_DETECTION_THRESHOLD=0.5,
)
