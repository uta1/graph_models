import sys
import requests
import tarfile
import json
import numpy as np
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob

MODULES = ['create_trg_image']
MODE = 'samples'
LABELS = MODE + '.json'
ARCHIVENAME = 'examples.tar.gz'
FOLDER = 'examples/'
BINFOLDER = FOLDER[:-1] + 'bin/'
URL = 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/' + ARCHIVENAME
MIN_OBJECT_WIDTH = 4
MIN_OBJECT_HEIGHT = 4

def get_file_names(folder):
    for images_info in os.walk(folder):
        return images_info[-1]

def download():
    r = requests.get(URL)
    open(ARCHIVENAME , 'wb').write(r.content)
    
def extract():
    tar = tarfile.open(ARCHIVENAME)
    tar.extractall()
    tar.close()
    
def get_samples():
    with open(FOLDER + LABELS, 'r') as fp:
        samples = json.load(fp)
    images = {}
    for image in samples['images']:
        images[image['id']] = {'file_name': FOLDER + image['file_name'], 'annotations': []}
    for ann in samples['annotations']:
        images[ann['image_id']]['annotations'].append(ann)
    return images

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def process():
    model = unet(input_size=(512,512,1))
    for image in get_file_names(BINFOLDER):
        img = cv2.imread(BINFOLDER + image)
        img = np.expand_dims(img[:,:,2:], axis=0)
        print(img.shape)
        model.predict(img)

import cv2
import os, time


def get_rects_by_contours(contours):
    rects = [cv2.boundingRect(contour) for contour in contours]
    return [rect for rect in rects if rect[-2] >= MIN_OBJECT_WIDTH and rect[-1] >= MIN_OBJECT_HEIGHT]


def create_trg_image(image_name, target_size=(512, 512), print_bboxes=False):
    if not image_name.startswith('PMC'):
        return

    original = cv2.imread(FOLDER + image_name)

    im_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)

    bined_resized = cv2.resize(im_gray_th_otsu, target_size, interpolation = cv2.INTER_NEAREST)

    if print_bboxes:
        contours, hier = cv2.findContours(bined_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        res_image = cv2.resize(original, target_size, interpolation = cv2.INTER_NEAREST)
        rects = get_rects_by_contours(contours)

        for x, y, w, h in rects:
            cv2.rectangle(res_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        res_image = bined_resized

    cv2.imwrite(BINFOLDER + 'trg_' + image_name + '.tiff', res_image)

if __name__ == '__main__':
    if 'download' in MODULES:
        download()
        extract()
        get_samples()
        print('downloading finished')
    if 'create_trg_image' in MODULES:
        for image in get_file_names(FOLDER):
            create_trg_image(image)
    if 'main' in MODULES:
        process()

