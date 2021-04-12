from lib_imports import *
from consts import *
from utils import *


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [list(rect) for rect in rects if rect[-2] >= MIN_OBJECT_WIDTH and rect[-1] >= MIN_OBJECT_HEIGHT]


def resize(orig, target_size):
    if target_size:
        return cv2.resize(orig, target_size, interpolation=cv2.INTER_NEAREST)
    return orig


def resize_rect(coef_width, coef_height, rect):
    return [
        math.floor(rect[0] * coef_width),
        math.floor(rect[1] * coef_height),
        math.floor(rect[2] * coef_width),
        math.floor(rect[3] * coef_height),
    ]


def resize_rects(target_size, orig_size, rects):
    if not target_size:
        return rects
    print(target_size, orig_size)

    coef_width = float(target_size[1]) / orig_size[1]
    coef_height = float(target_size[0]) / orig_size[0]
    return [resize_rect(coef_width, coef_height, rect) for rect in rects]


def create_trg_image(image_name, target_size=(512, 512), binarize=True, print_bboxes=False):
    if not image_name.startswith('PMC'):
        return

    original = cv2.imread(FOLDER + image_name)
    orig_size = original.shape[:2]
    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    th, bined = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECTS_DILATION)
    dilation = cv2.dilate(bined, rect_kernel, iterations=1)
    contours, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    resized_rects = resize_rects(target_size, orig_size, get_rects_by_contours(contours))

    filename_template =  'trg_' + image_name

    label_filename = LABELSFOLDER + filename_template + '.json'
    label = json.dumps(
        {
            'rects': resized_rects
        }
    )
    with open(label_filename, 'w') as fp:
        fp.write(label)

    res = resize(bined if binarize else original, target_size)
    if print_bboxes:
        if binarize:
            res = np.tile(res[..., None], 3)
        for x, y, w, h in resized_rects:
            cv2.rectangle(res, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
    filename = BINFOLDER + filename_template + '.tiff'
    cv2.imwrite(filename, res)
    print(res.shape, filename)

    return res


def create_trg_images():
    create_path(BINFOLDER)
    create_path(LABELSFOLDER)
    for image in get_file_names(FOLDER):
        create_trg_image(image, target_size=(512, 512), binarize=True, print_bboxes=False)

