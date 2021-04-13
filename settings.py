# Main
MODULES = ['prepare_samples', 'create_trg_images', 'train']
MODE = 'samples'

LABELS = MODE + '.json'


# Directories of data
SRCFOLDER = '../src/'
TRGFOLDER = '../trg/'

FOLDER = SRCFOLDER + MODE + '/'


# Directories of labels and target images
BINFOLDER = TRGFOLDER + MODE + 'bin/'
LABELSFOLDER = TRGFOLDER + MODE + 'labels/'
JSONLABELSFOLDER = TRGFOLDER + MODE + 'jsonlabels/'


# Labels creation
LABELSPATH = SRCFOLDER + LABELS
CACHEDLABELSPATH = JSONLABELSFOLDER + 'cached_' + LABELS
CACHEDIDBYFILENAMEPATH = JSONLABELSFOLDER + 'cached_id_by_file_name_' + LABELS


# Rects prediction
MIN_OBJECT_WIDTH = 4
MIN_OBJECT_HEIGHT = 4
RECTS_DILATION = (7, 6)


# Trg-creator settings
TARGET_SIZE = (512, 512)
BINARIZE = True
PLOT_BBOXES = None
FORCE_CACHE_CHECKING = True

