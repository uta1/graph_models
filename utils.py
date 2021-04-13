from lib_imports import *
from settings import *


def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]


def create_folder_if_not_exists(folder):
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)


def create_path(path):
    assert 'publaynet' not in path

    if path[-1] == '/':
        path = path[:-1]

    folders = path.split('/')

    cur_path = ''
    for folder in folders:
        cur_path += folder + '/'
        create_folder_if_not_exists(cur_path)


def cache_and_get_indices():
    if path.exists(CACHEDLABELSPATH):
        with open(CACHEDLABELSPATH, 'r') as fp:
            cached_labels = {
                int(image_id):value
                for image_id, value in json.load(fp).items()
            }
        with open(CACHEDIDBYFILENAMEPATH, 'r') as fp:
            image_id_by_file_name = json.load(fp)
    else:
        cached_labels, image_id_by_file_name = get_labels_indices()
        with open(CACHEDLABELSPATH, 'w') as fp:
            json.dump(cached_labels, fp=fp)
        with open(CACHEDIDBYFILENAMEPATH, 'w') as fp:
            json.dump(image_id_by_file_name, fp=fp)
    return cached_labels, image_id_by_file_name


def get_labels_indices():
    labels = get_labels_full()
    labels_by_image_id = {}
    image_id_by_file_name = {}
    for image in labels['images']:
        labels_by_image_id[image['id']] = {
            'file_name': FOLDER + image['file_name'],
            'bin_file_name': BINFOLDER + 'trg_' + image['file_name'],
            'labels_file_name': LABELSFOLDER + 'trg_' + image['file_name'],
            'annotations': []
        }
        image_id_by_file_name[image['file_name']] = image['id']
    for ann in labels['annotations']:
        labels_by_image_id[ann['image_id']]['annotations'].append(
            {
                'bbox': ann['bbox'],
                'category_id': ann['category_id']
            }
        )
    return labels_by_image_id, image_id_by_file_name


def get_labels_full():
    with open(LABELSPATH, 'r') as fp:
        return json.load(fp)

