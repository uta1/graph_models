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


def get_samples():
    with open(FOLDER + LABELS, 'r') as fp:
        samples = json.load(fp)
    images = {}
    for image in samples['images']:
        images[image['id']] = {'file_name': FOLDER + image['file_name'], 'annotations': []}
    for ann in samples['annotations']:
        images[ann['image_id']]['annotations'].append(ann)
    return images

