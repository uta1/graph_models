from lib_imports import *
from consts import *
from utils import *


ARCHIVENAME = 'examples.tar.gz'
URL = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + ARCHIVENAME
TARFOLDER = 'examples/'


def download():
    r = requests.get(URL)
    open(ARCHIVENAME , 'wb').write(r.content)


def extract():
    create_path(FOLDER)
    tar = tarfile.open(ARCHIVENAME)
    tar.extractall()
    tar.close()
    os.system('rm ' + ARCHIVENAME)
    os.system('mv ' + TARFOLDER + ' ' + DATASETFOLDER)

