# Enabled modules and mode
MODULES = ['prepare_samples', 'create_trg_images', 'main']
MODE = 'samples'

# Directory of initial dataset
DATASETTEMPLATE = 'trg'

# Directories of labels and target images
DATASETFOLDER = '../' + DATASETTEMPLATE + '/'
FOLDER = DATASETFOLDER + MODE + '/'
BINFOLDER = DATASETFOLDER + MODE + 'bin/'
LABELSFOLDER = DATASETFOLDER + MODE + 'labels/'

# Labels creation
LABELS = MODE + '.json'
LABELSPATH = DATASETFOLDER + LABELS
CACHEDLABELSPATH = LABELSFOLDER + 'cached_' + LABELS
CACHEDIDBYFILENAMEPATH = LABELSFOLDER + 'cached_id_by_file_name_' + LABELS

# Rects prediction
MIN_OBJECT_WIDTH = 4
MIN_OBJECT_HEIGHT = 4
RECTS_DILATION = (7, 6)

# Trg-creator settings
TARGET_SIZE = (512, 512)
BINARIZE = True
PLOT_BBOXES = None
FORCE_CACHE_CHECKING = True

