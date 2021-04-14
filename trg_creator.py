from lib_imports import *

from geometry import *
from config import *
from utils import *


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [
        list(rect)
        for rect in rects if (
                rect[-2] >= config.MIN_OBJECT_WIDTH and rect[-1] >= config.MIN_OBJECT_HEIGHT
        )
    ]


def extract_rects_from_metainfo(image_metainfo):
    return [
        ann['bbox']
        for ann in image_metainfo['annotations']
    ]


def prepare_bined(original):
    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def predict_rects(bined):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.RECTS_DILATION)
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


def build_label_by_rects(image_metainfo, shape):
    res = np.zeros(shape, dtype=np.int32)
    for ann in image_metainfo['annotations']:
        x, y, w, h = ann['bbox']
        res[y:y + h, x:x + w] = ann['category_id']
    return res


def prepare_trg_for_image(image_metainfo):
    original = cv2.imread(image_metainfo['file_path'])
    orig_size = original.shape[:2]

    bined = prepare_bined(original)
    predicted_rects = resize_rects(
        orig_size,
        predict_rects(bined)
    )
    # already resized in get_images_metainfo()
    labeled_rects = extract_rects_from_metainfo(image_metainfo)

    if config.SAVE_JSONS:
        save_image_json(image_metainfo, predicted_rects)

    res = resize(bined if config.BINARIZE else original)

    if config.BBOXES_TO_PLOT and config.BINARIZE:
        res = np.tile(res[..., None], 3)
    for bboxes_to_plot in config.BBOXES_TO_PLOT:
        color = get_color_to_plot_bboxes(bboxes_to_plot)
        for x, y, w, h in get_rects_to_plot_bboxes(bboxes_to_plot, predicted_rects, labeled_rects):
            cv2.rectangle(res, (x, y), (x + w, y + h), color=color, thickness=1)

    cv2.imwrite(
        image_metainfo['label_file_path'],
        build_label_by_rects(
            image_metainfo,
            shape=config.TARGET_SIZE or orig_size
        )
    )

    cv2.imwrite(image_metainfo['bin_file_path'], res)
    print(res.shape, image_metainfo['bin_file_path'])

    return res


def prepare_trg():
    create_path(config.BINS_FOLDER)
    create_path(config.LABELS_FOLDER)
    create_path(config.JSONS_FOLDER)

    images_metainfo = cache_and_get_images_metainfo()
    for metainfo in images_metainfo.values():
        prepare_trg_for_image(metainfo)
