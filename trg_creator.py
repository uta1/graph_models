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


def floor(rect):
    return [
        math.floor(elem)
        for elem in rect
    ]


def resize_rect(coef_width, coef_height, rect):
    return floor(
        [
            rect[0] * coef_width,
            rect[1] * coef_height,
            rect[2] * coef_width,
            rect[3] * coef_height,
        ]
    )


def resize_rects(target_size, orig_size, rects, force_floor=False):
    if not target_size:
        return [floor(rect) for rect in rects] if force_floor else rects

    coef_width = float(target_size[1]) / orig_size[1]
    coef_height = float(target_size[0]) / orig_size[0]
    return [resize_rect(coef_width, coef_height, rect) for rect in rects]


def extract_rects_from_label(image_name, cached_labels, image_id_by_file_name):
    return [
        ann['bbox']
        for ann in cached_labels[image_id_by_file_name[image_name]]['annotations']
    ]


# plot_bboxes = [None | 'predict' | 'labels']
def create_trg_image(
    image_name,
    cached_labels,
    image_id_by_file_name,
    target_size=(512, 512),
    binarize=True,
    plot_bboxes=None
):
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
    if plot_bboxes:
        if binarize:
            res = np.tile(res[..., None], 3)
        if plot_bboxes == 'labels':
            rects_to_plot = resize_rects(
                target_size,
                orig_size,
                extract_rects_from_label(image_name, cached_labels, image_id_by_file_name),
                force_floor=True
            )
        else:
            rects_to_plot = resized_rects
        for x, y, w, h in rects_to_plot:
            cv2.rectangle(res, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
    filename = BINFOLDER + filename_template + '.tiff'
    cv2.imwrite(filename, res)
    print(res.shape, filename)

    return res


def create_trg_images():
    target_size = (512, 512)
    binarize = True
    plot_bboxes = None

    create_path(BINFOLDER)
    create_path(LABELSFOLDER)
    cached_labels, image_id_by_file_name = get_labels_indices() if plot_bboxes == 'labels' else (None, None)
    for image in get_file_names(FOLDER):
        create_trg_image(
            image,
            cached_labels,
            image_id_by_file_name,
            target_size=target_size,
            binarize=binarize,
            plot_bboxes=plot_bboxes
        )

