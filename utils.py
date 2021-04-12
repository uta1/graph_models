from lib_imports import *

def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]

