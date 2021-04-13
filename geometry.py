from lib_imports import *
from settings import *


def resize(orig):
    if TARGET_SIZE:
        return cv2.resize(orig, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_NEAREST)
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


def floor_rects(rects):
    return [floor(rect) for rect in rects]


def resize_rects(orig_size, rects):
    if not TARGET_SIZE:
        return floor_rects(rects)

    coef_width, coef_height = get_resizing_coefs(orig_size[1], orig_size[0])
    return [resize_rect(coef_width, coef_height, rect) for rect in rects]

def get_resizing_coefs(orig_width, orig_height):
    if not TARGET_SIZE:
        return 1.0, 1.0

    return float(TARGET_SIZE[1]) / orig_width, float(TARGET_SIZE[0]) / orig_height
