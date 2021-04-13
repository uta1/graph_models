from lib_imports import *

from geometry import *
from settings import *
from utils import *


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [list(rect) for rect in rects if rect[-2] >= MIN_OBJECT_WIDTH and rect[-1] >= MIN_OBJECT_HEIGHT]


def extract_rects_from_label(image_name, cached_labels, image_id_by_file_name):
    return [
        ann['bbox']
        for ann in cached_labels[image_id_by_file_name[image_name]]['annotations']
    ]


def prepare_bined(original):
    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def predict_rects(bined):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECTS_DILATION)
    dilation = cv2.dilate(bined, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return get_rects_by_contours(contours)


def get_rects_to_plot_bboxes(bboxes_to_plot, predicted_rects, labeled_rects):
    if bboxes_to_plot == 'predicted':
        return predicted_rects
    if bboxes_to_plot == 'labeled':
        return labeled_rects


def get_color_to_plot_bboxes(bboxes_to_plot):
    if bboxes_to_plot == 'predicted':
        return (0, 255, 0)
    if bboxes_to_plot == 'labeled':
        return (255, 0, 255)


def save_image_json(image_name, predicted_rects):
    with open(image_name_to_json_path(image_name), 'w') as fp:
        fp.write(
            json.dumps(
                {
                    'rects': predicted_rects
                }
            )
        )


def build_label_by_rects(image_name, cached_labels, image_id_by_file_name, shape):
    res = np.zeros(shape, dtype=np.int32)
    for ann in cached_labels[image_id_by_file_name[image_name]]['annotations']:
        x, y, w, h = ann['bbox']
        res[y:y + h, x:x + w] = ann['category_id']
    return res


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
    predicted_rects = resize_rects(
        orig_size,
        predict_rects(bined)
    )
    # already resized in get_labels_indices()
    labeled_rects = extract_rects_from_label(image_name, cached_labels, image_id_by_file_name)

    if SAVE_JSONS:
        save_image_json(image_name, predicted_rects)

    res = resize(bined if BINARIZE else original)

    if BBOXES_TO_PLOT and BINARIZE:
        res = np.tile(res[..., None], 3)
    for bboxes_to_plot in BBOXES_TO_PLOT:
        color = get_color_to_plot_bboxes(bboxes_to_plot)
        for x, y, w, h in get_rects_to_plot_bboxes(bboxes_to_plot, predicted_rects, labeled_rects):
            cv2.rectangle(res, (x, y), (x + w, y + h), color=color, thickness=1)

    cv2.imwrite(
        image_name_to_label_path(image_name),
        build_label_by_rects(
            image_name,
            cached_labels,
            image_id_by_file_name,
            shape=TARGET_SIZE or orig_size
        )
    )

    bin_file_name = image_name_to_bin_path(image_name)
    cv2.imwrite(bin_file_name, res)
    print(res.shape, bin_file_name)

    return res


def prepare_trg():
    create_path(BINS_FOLDER)
    create_path(LABELS_FOLDER)
    create_path(JSONS_FOLDER)

    cached_labels, image_id_by_file_name = (None, None)
    if BBOXES_TO_PLOT == 'labeled' or FORCE_CACHE_CHECKING:
        cached_labels, image_id_by_file_name = cache_and_get_indices()

    for image_name in get_file_names(FOLDER):
        prepare_trg_for_image(
            image_name,
            cached_labels,
            image_id_by_file_name,
        )
