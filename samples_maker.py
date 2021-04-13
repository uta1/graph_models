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
    if not path.exists(FOLDER):
        create_path(SRC_FOLDER)
        tar = tarfile.open(ARCHIVENAME)
        tar.extractall()
        tar.close()
        os.system('mv ' + TARFOLDER + ' ' + SRC_FOLDER)
        os.system('mv ' + (SRC_FOLDER + TARFOLDER) + ' ' + FOLDER)
        if path.exists(FOLDER + LABELS):
            os.system('mv ' + (FOLDER + LABELS) + ' ' + LABELS_PATH)
    os.system('rm ' + ARCHIVENAME)


def prepare_samples():
    # for safety
    if MODE != 'samples' or 'publaynet' in SRC_FOLDER:
        return

    download()
    extract()
