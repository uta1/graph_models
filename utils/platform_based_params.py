from lib_imports import *


def folders_delim():
    if platform.system() == 'Linux':
        return '/'
    return '\\'


def workplace_dir():
    if platform.system() == 'Linux':
        return '../'
    raise
