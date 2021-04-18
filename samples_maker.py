from lib_imports import *

from config import config
from utils.filesystem_helper import (
    create_path,
    folder_path,
    labels_file_name,
    labels_file_path,
    src_folder_path,
)

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
        not os.path.exists(folder_path('train'))
    ):
        create_path(src_folder_path())

        tar = tarfile.open(ARCHIVENAME)
        tar.extractall()
        tar.close()

        os.system('mv ' + TARFOLDER + ' ' + src_folder_path())
        os.system('mv ' + (src_folder_path() + TARFOLDER) + ' ' + folder_path('train'))
        if os.path.exists(folder_path('train') + labels_file_name('train')):
            os.system(
                'mv ' +
                (folder_path('train') + labels_file_name('train')) +
                ' ' +
                labels_file_path('train')
            )

        if config.VAL_FOLDER_NAME != config.TRAIN_FOLDER_NAME:
            os.system('cp -r ' + folder_path('train') + ' ' + folder_path('val'))
            os.system('cp ' + labels_file_path('train') + ' ' + labels_file_path('val'))

    os.system('rm ' + ARCHIVENAME)


def prepare_samples():
    # for safety
    if config.TRAIN_FOLDER_NAME != 'samples' or 'publaynet' in src_folder_path():
        return

    download()
    extract()
