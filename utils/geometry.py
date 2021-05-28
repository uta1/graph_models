from lib_imports import *

from config import *


def _floor(rect):
    return [
        math.floor(elem)
        for elem in rect
    ]


def _floor_rects(rects):
    return [_floor(rect) for rect in rects]


def resize(orig):
    if config.TARGET_SIZE:
        return cv2.resize(
            orig,
            (config.TARGET_SIZE[1], config.TARGET_SIZE[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return orig


def resize_rect(coef_width, coef_height, rect):
    return _floor(
        [
            rect[0] * coef_width,
            rect[1] * coef_height,
            rect[2] * coef_width,
            rect[3] * coef_height,
        ]
    )


def resize_rects(orig_size, rects):
    if not config.TARGET_SIZE:
        return _floor_rects(rects)

    coef_width, coef_height = get_resizing_coefs(orig_size[1], orig_size[0])
    return [resize_rect(coef_width, coef_height, rect) for rect in rects]


def get_resizing_coefs(orig_width, orig_height):
    if not config.TARGET_SIZE:
        return 1.0, 1.0

    return float(config.TARGET_SIZE[1]) / orig_width, float(config.TARGET_SIZE[0]) / orig_height


def rects_intersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    left = max(x1, x2)
    top = min(y1 + h1, y2 + h2)
    right = min(x1 + w1, x2 + w2)
    bottom = max(y1, y2)

    width = right - left
    height = top - bottom

    if width < 0.0 or height < 0.0:
        return 0.0

    return width * height
