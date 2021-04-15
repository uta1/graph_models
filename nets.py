from lib_imports import *

from config import *
from logger import *


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # Original is commented
    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    conv9 = Conv2D(6, 3, activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)

    rois = tf.convert_to_tensor(
        [[[10, 10, 10, 10], [10, 10, 10, 10]]],
        dtype=tf.float32
    )
    outputs = []
    for roi_idx in range(len(rois[0])):

        x = rois[0, roi_idx, 0]
        y = rois[0, roi_idx, 1]
        w = rois[0, roi_idx, 2]
        h = rois[0, roi_idx, 3]

        x = K.cast(x, 'int32')
        y = K.cast(y, 'int32')
        w = K.cast(w, 'int32')
        h = K.cast(h, 'int32')

        roi = Lambda(lambda arg: arg[:, y:y + h, x:x + w, :], name="Lambda_" + str(roi_idx))(conv9)
        resized_roi = Lambda(lambda image: tf.image.resize(image, (16, 16)), name="Lambda2_" + str(roi_idx))(roi)

        outputs.append(resized_roi)

    concated_rois = Concatenate(axis=0, name='concat_rois')(outputs)

    model = Model(inputs=inputs, outputs=concated_rois)

    model.trainable = config.is_model_trainable()

    model.compile(
        optimizer=Adam(lr=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', IOUScore(name='iou_score')]
    )

    model.summary(print_fn=lambda x: logger.log(x))

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
