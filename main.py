from lib_imports import *
from consts import *
from downloader import *
from utils import *
from nets import *


def process():
    model = unet(input_size=(512,512,1))
    for image in get_file_names(BINFOLDER):
        img = cv2.imread(BINFOLDER + image)
        img = np.expand_dims(img[:,:,2:], axis=0)
        print(img.shape)
        model.predict(img)


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [rect for rect in rects if rect[-2] >= MIN_OBJECT_WIDTH and rect[-1] >= MIN_OBJECT_HEIGHT]


def resize(orig, target_size):
    if target_size:
        return cv2.resize(orig, target_size, interpolation=cv2.INTER_NEAREST)
    return orig


def create_trg_image(image_name, target_size=(512, 512), print_bboxes=True):
    if not image_name.startswith('PMC'):
        return

    original = cv2.imread(FOLDER + image_name)

    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    th, bined = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    if print_bboxes:
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, RECTS_DILATION)
        dilation = cv2.dilate(bined, rect_kernel, iterations=1)

        contours, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        rects = get_rects_by_contours(contours)

        res_not_resized = original
        for x, y, w, h in rects:
            cv2.rectangle(res_not_resized, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
    else:
        res_not_resized = bined

    res = resize(res_not_resized, target_size)

    filename = BINFOLDER + 'trg_' + image_name + '.tiff'
    cv2.imwrite(filename, res)
    print(res.shape, filename)
    return res


def create_trg_images():
    create_folder_if_not_exists(BINFOLDER)
    for image in get_file_names(FOLDER):
        create_trg_image(image, target_size=(512, 512), print_bboxes=False)


if __name__ == '__main__':
    if 'download' in MODULES:
        download()
        extract()
        get_samples()

    if 'create_trg_image' in MODULES:
        create_trg_images()

    if 'main' in MODULES:
        process()

