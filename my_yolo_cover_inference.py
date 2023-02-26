import os
import cv2 as cv
import csv
import numpy as np
from icecream import ic
import tensorflow as tf
import glob
import cfg
import time
import os.path as op
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, concatenate, Activation

MODEL_INPUT_SIZE = cfg.MODEL_INPUT_SIZE
MODEL_W = float(cfg.MODEL_INPUT_SIZE[1])
MODEL_H = float(cfg.MODEL_INPUT_SIZE[0])
DIV = cfg.DIV

# EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3),
    pooling=None,
)

yolo = Conv2D(192, 1, activation=None, padding="same")(backbone.layers[-1].output)
yolo = BatchNormalization()(yolo)
yolo = LeakyReLU()(yolo)
yolo = UpSampling2D()(yolo)

tap_16 = Conv2D(192, 1, activation=None, padding="same")(backbone.get_layer('block6a_expand_activation').output)
tap_16 = BatchNormalization()(tap_16)
tap_16 = LeakyReLU()(tap_16)

yolo = concatenate([tap_16, yolo])    # 192+192

yolo = Conv2D(256, 3, activation=None, padding="same")(yolo)
yolo = BatchNormalization()(yolo)
yolo = LeakyReLU()(yolo)
yolo = Conv2D(5, 1, activation=None, padding="same", name='YOLO')(yolo)
yolo = Activation(tf.keras.activations.linear)(yolo)
model = tf.keras.Model(inputs=backbone.inputs, outputs=yolo)

model.compile()

model.load_weights(cfg.MODEL_NAME)


# Generate test set
#files = glob.glob(R'F:\crowd-selected\test\*.png')
files = glob.glob(R'F:\crowd-selected\test\small\*.jpg')
for idx, img_fn in enumerate(files):
    img = cv.imread(img_fn)
    img = tf.image.resize(img,(MODEL_INPUT_SIZE[0],MODEL_INPUT_SIZE[1]))
    xdata = tf.expand_dims(img, axis=0)
    preds = model(xdata[0:1,...], training=True).numpy()
    preds = preds[0,...]
    img_cv = xdata[0,...]/255
    img_cv = img_cv.numpy()
    for j in range(preds.shape[0]):        # h
        for k in range(preds.shape[1]):    # w                
            img_cv[DIV//2+j*DIV, DIV//2+k*DIV, :] = [1,0,0]      # illustrate grid center
            conf = preds[j,k,0]                            # illustrate prediction
            if conf > 0.5:
                box_xy = preds[j,k,1:3]                  # range 0-1
                box_xy  = DIV * (2 * box_xy - [1.,1.])
                grid_cen_xy = np.array([DIV//2+k*DIV, DIV//2+j*DIV])
                box_xy  = grid_cen_xy + box_xy
                box_wh = preds[j,k,3:5] * np.array([MODEL_W,MODEL_H]) * 0.25      # range 0-1
                pt0 = box_xy - box_wh/2
                pt1 = box_xy + box_wh/2
                cv.rectangle(img_cv, pt0.astype(int), pt1.astype(int), (0,0,1), 1)

    out_name = op.splitext(op.basename(img_fn))[0] + '_output.png'
    cv.imwrite(op.join(cfg.OUTPUT_DIR, out_name), img_cv*255)

print('Completed')
