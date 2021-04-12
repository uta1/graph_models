from lib_imports import *
from consts import *


def download():
    r = requests.get(URL)
    open(ARCHIVENAME , 'wb').write(r.content)
    
def extract():
    tar = tarfile.open(ARCHIVENAME)
    tar.extractall()
    tar.close()
    os.system('rm ' + ARCHIVENAME)
    os.system('mv ' + TARFOLDER + ' ' + FOLDER)
    
def get_samples():
    with open(FOLDER + LABELS, 'r') as fp:
        samples = json.load(fp)
    images = {}
    for image in samples['images']:
        images[image['id']] = {'file_name': FOLDER + image['file_name'], 'annotations': []}
    for ann in samples['annotations']:
        images[ann['image_id']]['annotations'].append(ann)
    return images

