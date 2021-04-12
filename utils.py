from lib_imports import *

def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]

def create_folder_if_not_exists(folder):
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)

