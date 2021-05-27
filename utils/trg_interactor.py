from lib_imports import *

from config import config
from utils.geometry import resize, resize_rects, resize_rect, get_resizing_coefs
from utils.filesystem_helper import (
    bins_folder_path,
    cached_labels_path,
    create_path,
    image_name_to_path,
    image_name_to_bin_path,
    image_name_to_label_path,
    jsons_folder_path,
    labels_file_path,
    labels_folder_path,
)


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


def cache_and_get_images_metainfo(mode):
    if os.path.exists(cached_labels_path(mode)):
        with open(cached_labels_path(mode), 'r') as fp:
            images_metainfo = {
                int(image_id): value
                for image_id, value in json.load(fp).items()
            }
    else:
        images_metainfo = _get_images_metainfo(mode)
        with open(cached_labels_path(mode), 'w') as fp:
            json.dump(images_metainfo, fp=fp)
    return images_metainfo


def _get_images_metainfo(mode):
    labels = _get_images_fullinfo(mode)
    images_metainfo = {}
    dims_by_image_id = {}
    for image in labels['images']:
        images_metainfo[image['id']] = {
            'file_path': image_name_to_path(mode, image['file_name']),
            'bin_file_path': image_name_to_bin_path(mode, image['file_name']),
            'label_file_path': image_name_to_label_path(mode, image['file_name']),
            'annotations': [],
            'rois': []
        }
        dims_by_image_id[image['id']] = {
            'width': image['width'],
            'height': image['height']
        }

    print('1st stage of trg creating done')

    for ann in labels['annotations']:
        image_id = ann['image_id']
        coef_width, coef_height = get_resizing_coefs(
            dims_by_image_id[image_id]['width'],
            dims_by_image_id[image_id]['height']
        )
        images_metainfo[image_id]['annotations'].append(
            {
                'bbox': resize_rect(coef_width, coef_height, ann['bbox']),
                'category_id': ann['category_id']
            }
        )

    print('2nd stage of trg creating done')

    total = len(images_metainfo.keys())
    for ind, image_id in enumerate(images_metainfo.keys()):
        image_metainfo = images_metainfo[image_id]

        original = cv2.imread(image_metainfo['file_path'])
        orig_size = original.shape[:2]

        bined = _prepare_bined(original)

        res = resize(bined if config.BINARIZE else original)

        if config.BBOXES_TO_PLOT and config.BINARIZE:
            res = np.tile(res[..., None], 3)
        predicted_rects = resize_rects(
            orig_size,
            _predict_rects(bined)
        )
        image_metainfo['rois'] = predicted_rects
        for bboxes_to_plot in config.BBOXES_TO_PLOT:
            color = _get_color_to_plot_bboxes(bboxes_to_plot)

            # already resized previously
            labeled_rects = _extract_rects_from_metainfo(image_metainfo)
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

        images_metainfo[image_id] = image_metainfo

        print(f'{ind + 1}/{total} images processed for {mode}')

    return images_metainfo


def _get_images_fullinfo(mode):
    with open(labels_file_path(mode), 'r') as fp:
        return json.load(fp)


def prepare_trg():
    for mode in config.MODES_TO_CREATE_TRG:
        create_path(bins_folder_path(mode))
        create_path(labels_folder_path(mode))
        create_path(jsons_folder_path(mode))

        cache_and_get_images_metainfo(mode)
