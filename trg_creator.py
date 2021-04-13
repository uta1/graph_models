from lib_imports import *
from settings import *
from utils import *


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [list(rect) for rect in rects if rect[-2] >= MIN_OBJECT_WIDTH and rect[-1] >= MIN_OBJECT_HEIGHT]


def resize(orig):
    if TARGET_SIZE:
        return cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
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


def resize_rects(orig_size, rects, force_floor=False):
    if not TARGET_SIZE:
        return [floor(rect) for rect in rects] if force_floor else rects

    coef_width = float(TARGET_SIZE[1]) / orig_size[1]
    coef_height = float(TARGET_SIZE[0]) / orig_size[0]
    return [resize_rect(coef_width, coef_height, rect) for rect in rects]


def extract_rects_from_label(image_name, cached_labels, image_id_by_file_name):
    return [
        ann['bbox']
        for ann in cached_labels[image_id_by_file_name[image_name]]['annotations']
    ]


def prepare_bined(original):
    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def prepare_predicted_rects(bined, orig_size):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECTS_DILATION)
    dilation = cv2.dilate(bined, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return resize_rects(orig_size, get_rects_by_contours(contours))


def prepare_trg_for_image(
        image_name,
        cached_labels,
        image_id_by_file_name,
):
    if not image_name.startswith('PMC'):
        return

    original = cv2.imread(FOLDER + image_name)
    orig_size = original.shape[:2]

    bined = prepare_bined(original)
    predicted_rects = prepare_predicted_rects(bined, orig_size)

    if SAVE_JSONS:
        with open(image_name_to_json_path(image_name), 'w') as fp:
            fp.write(
                json.dumps(
                    {
                        'rects': predicted_rects
                    }
                )
            )

    res = resize(bined if BINARIZE else original)
    if PLOT_BBOXES:
        if BINARIZE:
            res = np.tile(res[..., None], 3)
        if PLOT_BBOXES == 'labels':
            rects_to_plot = resize_rects(
                orig_size,
                extract_rects_from_label(image_name, cached_labels, image_id_by_file_name),
                force_floor=True
            )
        else:
            rects_to_plot = predicted_rects
        for x, y, w, h in rects_to_plot:
            cv2.rectangle(res, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
    bin_file_name = image_name_to_bin_path(image_name)
    cv2.imwrite(bin_file_name, res)
    print(res.shape, bin_file_name)

    return res


def prepare_trg():
    create_path(BINS_FOLDER)
    create_path(LABELS_FOLDER)
    create_path(JSONS_FOLDER)

    cached_labels, image_id_by_file_name = (None, None)
    if PLOT_BBOXES == 'labels' or FORCE_CACHE_CHECKING:
        cached_labels, image_id_by_file_name = cache_and_get_indices()

    for image_name in get_file_names(FOLDER):
        prepare_trg_for_image(
            image_name,
            cached_labels,
            image_id_by_file_name,
        )
