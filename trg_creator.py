from lib_imports import *

from config import config
from utils.geometry import resize, resize_rects
from utils.filesystem_helper import create_path, bins_folder_path, labels_folder_path, jsons_folder_path
from utils.images_metainfo_cacher import cache_and_get_images_metainfo


def _get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [
        list(rect)
        for rect in rects if (
                rect[-2] >= config.MIN_OBJECT_WIDTH and rect[-1] >= config.MIN_OBJECT_HEIGHT
        )
    ]


def _extract_rects_from_metainfo(image_metainfo):
    return [
        ann['bbox']
        for ann in image_metainfo['annotations']
    ]


def _prepare_bined(original):
    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]


def _predict_rects(bined):
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.RECTS_DILATION)
    dilation = cv2.dilate(bined, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return _get_rects_by_contours(contours)


def _get_rects_to_plot_bboxes(bboxes_to_plot, predicted_rects, labeled_rects):
    if bboxes_to_plot == 'predicted':
        return predicted_rects
    if bboxes_to_plot == 'labeled':
        return labeled_rects


def _get_color_to_plot_bboxes(bboxes_to_plot):
    if bboxes_to_plot == 'predicted':
        return (0, 255, 0)
    if bboxes_to_plot == 'labeled':
        return (255, 0, 255)


def _build_label_by_rects(image_metainfo, shape):
    res = np.zeros(shape, dtype=np.int32)
    for ann in image_metainfo['annotations']:
        x, y, w, h = ann['bbox']
        res[y:y + h, x:x + w] = ann['category_id']
    return res


def _prepare_trg_for_image(image_metainfo):
    original = cv2.imread(image_metainfo['file_path'])
    orig_size = original.shape[:2]

    bined = _prepare_bined(original)
    # already resized in _get_images_metainfo
    labeled_rects = _extract_rects_from_metainfo(image_metainfo)

    res = resize(bined if config.BINARIZE else original)

    if config.BBOXES_TO_PLOT and config.BINARIZE:
        res = np.tile(res[..., None], 3)
    for bboxes_to_plot in config.BBOXES_TO_PLOT:
        color = _get_color_to_plot_bboxes(bboxes_to_plot)
        predicted_rects = resize_rects(
            orig_size,
            _predict_rects(bined)
        )
        for x, y, w, h in _get_rects_to_plot_bboxes(bboxes_to_plot, predicted_rects, labeled_rects):
            cv2.rectangle(res, (x, y), (x + w, y + h), color=color, thickness=1)

    cv2.imwrite(
        image_metainfo['label_file_path'],
        _build_label_by_rects(
            image_metainfo,
            shape=config.TARGET_SIZE or orig_size
        )
    )

    cv2.imwrite(image_metainfo['bin_file_path'], res)

    return res


def prepare_trg():
    for mode in config.MODES_TO_CREATE_TRG:
        create_path(bins_folder_path(mode))
        create_path(labels_folder_path(mode))
        create_path(jsons_folder_path(mode))

        images_metainfo = cache_and_get_images_metainfo(mode)
        total = len(images_metainfo)
        for ind, metainfo in enumerate(images_metainfo.values()):
            _prepare_trg_for_image(metainfo)
            print(f'{ind}/{total} images processed for {mode}')
