from lib_imports import *

from config import config
from utils.filesystem_helper import create_path

ARCHIVENAME = 'examples.tar.gz'
URL = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + ARCHIVENAME
TARFOLDER = 'examples/'


def download():
    r = requests.get(URL)
    open(ARCHIVENAME, 'wb').write(r.content)


def extract():
    if (
        config.TRAIN_FOLDER_NAME == 'samples' and
        config.SRC_FOLDER_NAME == 'src' and
        not os.path.exists(config.folder_path('train'))
    ):
        create_path(config.SRC_FOLDER_PATH)

        tar = tarfile.open(ARCHIVENAME)
        tar.extractall()
        tar.close()

        os.system('mv ' + TARFOLDER + ' ' + config.SRC_FOLDER_PATH)
        os.system('mv ' + (config.SRC_FOLDER_PATH + TARFOLDER) + ' ' + config.folder_path('train'))
        if os.path.exists(config.folder_path('train') + config.labels_file_name('train')):
            os.system(
                'mv ' +
                (config.folder_path('train') + config.labels_file_name('train')) +
                ' ' +
                config.labels_file_path('train')
            )

        if config.VAL_FOLDER_NAME != config.TRAIN_FOLDER_NAME:
            os.system('cp -r ' + config.folder_path('train') + ' ' + config.folder_path('val'))
            os.system('cp ' + config.labels_file_path('train') + ' ' + config.labels_file_path('val'))

    os.system('rm ' + ARCHIVENAME)


def prepare_samples():
    # for safety
    if config.TRAIN_FOLDER_NAME != 'samples' or 'publaynet' in config.SRC_FOLDER_PATH:
        return

    download()
    extract()
