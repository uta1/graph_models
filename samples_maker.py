from lib_imports import *

from settings import *
from utils import *

ARCHIVENAME = 'examples.tar.gz'
URL = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + ARCHIVENAME
TARFOLDER = 'examples/'


def download():
    r = requests.get(URL)
    open(ARCHIVENAME, 'wb').write(r.content)


def extract():
    if not os.path.exists(config.FOLDER):
        create_path(config.SRC_FOLDER)
        tar = tarfile.open(ARCHIVENAME)
        tar.extractall()
        tar.close()
        os.system('mv ' + TARFOLDER + ' ' + config.SRC_FOLDER)
        os.system('mv ' + (config.SRC_FOLDER + TARFOLDER) + ' ' + config.FOLDER)
        if os.path.exists(config.FOLDER + config.LABELS):
            os.system('mv ' + (config.FOLDER + config.LABELS) + ' ' + config.LABELS_PATH)
    os.system('rm ' + ARCHIVENAME)


def prepare_samples():
    # for safety
    if config.MODE != 'samples' or 'publaynet' in config.SRC_FOLDER:
        return

    download()
    extract()
