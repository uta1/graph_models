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


def resize_rect(target_size, orig_size, rect):
    coef_width = float(target_size[0]) / orig_size[0]
    coef_height = float(target_size[1]) / orig_size[1]

    return [
        math.floor(rect[0] * coef_width),
        math.floor(rect[1] * coef_height),
        math.floor(rect[2] * coef_width),
        math.floor(rect[3] * coef_height),
    ]


def create_json_label(target_size, orig_size, rects):
    return {
        'rects': (
            rects if not target_size else
            [resize_rect(target_size, orig_size, rect) for rect in rects]
        )
    }


def create_trg_image(image_name, target_size=(512, 512), print_bboxes=True):
    if not image_name.startswith('PMC'):
        return

    original = cv2.imread(FOLDER + image_name)

    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    th, bined = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECTS_DILATION)
    dilation = cv2.dilate(bined, rect_kernel, iterations=1)
    contours, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = get_rects_by_contours(contours)

    if print_bboxes:
        res_not_resized = original
        for x, y, w, h in rects:
            cv2.rectangle(res_not_resized, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
    else:
        res_not_resized = bined

    res = resize(res_not_resized, target_size)

    filename_template =  'trg_' + image_name
    filename = BINFOLDER + filename_template + '.tiff'
    cv2.imwrite(filename, res)
    print(res.shape, filename)

    label_filename = LABELSFOLDER + filename_template + '.json'
    label = json.dumps(create_json_label(target_size, original.shape[:2], rects))
    with open(label_filename, 'w') as fp:
        fp.write(label)

    return res


def create_trg_images():
    create_path(BINFOLDER)
    create_path(LABELSFOLDER)
    for image in get_file_names(FOLDER):
        create_trg_image(image, target_size=(512, 512), print_bboxes=False)

