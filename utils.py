from lib_imports import *
from consts import *


def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def create_folder_if_not_exists(folder):
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)


def create_path(path):
    if path[-1] == '/':
        path = path[:-1]

    folders = path.split('/')
    
    cur_path = ''
    for folder in folders:
        cur_path += folder + '/'
        create_folder_if_not_exists(cur_path)


def get_samples():
    with open(FOLDER + LABELS, 'r') as fp:
        samples = json.load(fp)
    images = {}
    for image in samples['images']:
        images[image['id']] = {'file_name': FOLDER + image['file_name'], 'annotations': []}
    for ann in samples['annotations']:
        images[ann['image_id']]['annotations'].append(ann)
    return images

