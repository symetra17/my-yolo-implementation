import cv2 as cv
from icecream import ic
import os
import cfg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger
from jpg2npy import images_to_numpy
from cut_tag import cut_tag

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, concatenate, Activation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


MODEL_W = float(cfg.MODEL_INPUT_SIZE[1])
MODEL_H = float(cfg.MODEL_INPUT_SIZE[0])


class Dynamo(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size):
        self.reload_from_disk()
        self.batch_size = batch_size

    def __getitem__(self, index):
            p = index
            batch_x = self.xdata[self.batch_size*p: self.batch_size*(p+1),...].copy()    #  + np.random.normal(loc=0.0, scale=4.0, size=(1,int(MODEL_H),int(MODEL_W),3) )
            batch_y = self.ydata[self.batch_size*p: self.batch_size*(p+1),...].copy()
            return batch_x, batch_y

    def __len__(self):
        return self.xdata.shape[0] // self.batch_size

    def reload_from_disk(self):
        self.xdata = np.load('xdata.npy')
        self.ydata = np.load('ydata.npy')


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, generatr_handle):
        self.n_history = 0
        self.generatr_handle = generatr_handle

    def on_epoch_end(self, batch, logs=None):
        if self.n_history % 50 == 49:
            print('\nSaving model\n')
            self.model.save_weights(cfg.MODEL_NAME)

        cut_tag(AUGM=cfg.GEO_AUG)
        images_to_numpy(random_flip=True, random_order=True, COLOR_AUG=cfg.COLOR_AUG, BRIGHTNESS_AUG=cfg.BRIGHTNESS_AUG)
        print('\nReload from disk')
        self.generatr_handle.reload_from_disk()
        self.n_history += 1
        if not os.path.exists('continue'):
            quit()


def iou_pix(pred_mins, pred_maxes, true_mins, true_maxes):
    # In pixel uint
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / (union_areas+1e-2)
    return iou_scores



def ciou_pix(pred_mins, pred_maxes, true_mins, true_maxes):
    # Complete-IOU loss, (CIOU), normalized distance, and aspect ratio
    # In pixel uint
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union = pred_areas + true_areas - intersect
    union = tf.maximum(union, 1e-4)
    iou_scores = intersect / union

    big_maxes = tf.maximum(pred_maxes, true_maxes)
    big_mins = tf.minimum(pred_mins, true_mins)
    big_wh = big_maxes - big_mins

    diag_sq = tf.square(big_wh[...,0]) +  tf.square(big_wh[...,1])
    
    pred_centr = (pred_maxes + pred_mins)/2.
    true_centr = (true_maxes + true_mins)/2.
    d_center_sq = tf.square(pred_centr-true_centr)
    center_dist_sq = d_center_sq[...,0] + d_center_sq[...,1]

    aspect_err = tf.math.atan(true_wh[...,0]/(true_wh[...,1]+1e-4)) - tf.math.atan(pred_wh[...,0]/(pred_wh[...,1]+1e-4))
    aspect_err = 0.4*tf.square(aspect_err)

    #diou_scores = iou_scores - center_dist_sq / (diag_sq+1e-4)  - 0.5 * aspect_err

    diag_sq = tf.maximum(diag_sq, 1e-4)      # avoid divided by zero error

    scores = iou_scores - center_dist_sq / diag_sq
    return scores

DIV = cfg.DIV
fDIV = float(DIV)

def compute_iou(pred_xy, pred_wh, true_xy, true_wh):
    # convert 0-1 range to pixel unit
    true_xy_pix = (2.*true_xy-1.) * fDIV
    true_wh_pix = true_wh * tf.constant([MODEL_H,MODEL_W]) * 0.25
    pred_xy_pix = (2.*pred_xy-1.) * fDIV
    pred_wh_pix = pred_wh * tf.constant([MODEL_H,MODEL_W]) * 0.25
    true_mins = true_xy_pix - true_wh_pix/2
    true_maxs = true_xy_pix + true_wh_pix/2
    pred_mins = pred_xy_pix - pred_wh_pix/2
    pred_maxs = pred_xy_pix + pred_wh_pix/2
    ious = iou_pix(pred_mins, pred_maxs, true_mins, true_maxs)
    return ious

def compute_ciou(pred_xy, pred_wh, true_xy, true_wh):
    # convert 0-1 range to pixel unit
    true_xy_pix = (2.*true_xy-1.) * fDIV
    true_wh_pix = true_wh * tf.constant([MODEL_H,MODEL_W]) * 0.25
    pred_xy_pix = (2.*pred_xy-1.) * fDIV
    pred_wh_pix = pred_wh * tf.constant([MODEL_H,MODEL_W]) * 0.25
    true_mins = true_xy_pix - true_wh_pix/2
    true_maxs = true_xy_pix + true_wh_pix/2
    pred_mins = pred_xy_pix - pred_wh_pix/2
    pred_maxs = pred_xy_pix + pred_wh_pix/2
    ious = ciou_pix(pred_mins, pred_maxs, true_mins, true_maxs)
    return ious


def compute_box_loss(true_xy, pred_xy, true_wh, pred_wh, has_obj_mask):
    weighting_xy = 5.1
    weighting_wh = 2.1
    xy_loss = tf.square(true_xy-pred_xy)
    xy_loss = tf.reduce_sum(xy_loss, axis=-1)      # combined x loss and y loss into one loss
    xy_loss = xy_loss * has_obj_mask               # apply is-object mask
    xy_loss = weighting_xy * tf.reduce_sum(xy_loss)          # combine all grid cell into a single loss value    

    wh_loss = tf.square(true_wh - pred_wh)    # tf.square(tf.sqrt(true_wh) - tf.sqrt(pred_wh))
    wh_loss = tf.reduce_sum(wh_loss, axis=-1)        # combine w and h  loss in one value
    wh_loss = wh_loss * has_obj_mask                 # apply has-object mask
    wh_loss = weighting_wh * tf.reduce_sum(wh_loss)            # combine all grid cell into a single loss value
    return xy_loss + wh_loss


def my_loss(y_true, y_pred):
    # one batch_size y_true.shape and y_pred.shape    
    true_conf = y_true[...,0]
    pred_conf = tf.keras.activations.sigmoid( y_pred[...,0] )

    obj_conf_loss = tf.abs(true_conf - pred_conf) * true_conf
    obj_conf_loss = tf.reduce_sum(obj_conf_loss)

    noobj_conf_loss = tf.abs(true_conf - pred_conf) * (1.-true_conf)
    noobj_conf_loss = tf.reduce_sum(noobj_conf_loss)

    # since many grid cells in an image don’t contain any objects, the 
    # sum-squared error tries to make the confidence score of these 
    # cells to zero. This means that their loss will dominate the 
    # gradients and it doesn’t let the model converge. To solve this 
    # issue, the authors introduce the parameters λcoord and λnoobj.
    #bkgnd_weight = .5
    bkgnd_weight = .8
    conf_loss = obj_conf_loss + bkgnd_weight * noobj_conf_loss       # just a little compensate for class imbalance
    true_xy = y_true[...,1:3]
    pred_xy = y_pred[...,1:3]
    true_wh = y_true[...,3:5]
    pred_wh = y_pred[...,3:5]
    iou_loss = 1. - compute_ciou(pred_xy, pred_wh, true_xy, true_wh)
    iou_loss = tf.reduce_sum( iou_loss * true_conf)

    return (conf_loss + iou_loss) / 128.
    #return conf_loss / 128.


def recall(y_true, y_pred):
    # one batch size
    conf_true = y_true[...,0]
    conf_pred = tf.keras.activations.sigmoid( y_pred[...,0] )     #conf_pred = y_pred[...,0]
    conf_pred = tf.cast(conf_pred>=.5, tf.float32)
    n_good_pred = tf.reduce_sum(conf_true*conf_pred)
    return n_good_pred / (tf.reduce_sum(conf_true) + 1e-5)


def good_pred(y_true, y_pred):
    conf_true = y_true[...,0]
    conf_pred = y_pred[...,0]
    conf_pred = tf.cast(conf_pred>=.5, tf.float32)
    n_good = tf.reduce_sum(conf_true*conf_pred)
    return n_good

def pre(y_true, y_pred):
    conf_true = y_true[...,0]
    conf_pred = tf.keras.activations.sigmoid( y_pred[...,0] )
    conf_pred = tf.cast(conf_pred>=.5, tf.float32)
    n_good_pred = tf.reduce_sum(conf_true*conf_pred)
    return n_good_pred / (tf.reduce_sum(conf_pred) + 1e-5)

def ave_iou(y_true, y_pred):
    true_conf = y_true[...,0]
    true_xy = y_true[...,1:3]
    pred_xy = y_pred[...,1:3]
    true_wh = y_true[...,3:5]
    pred_wh = y_pred[...,3:5]
    ious = compute_iou(pred_xy, pred_wh, true_xy, true_wh)
    iou = tf.reduce_sum( ious * true_conf )
    return iou/ (tf.reduce_sum(true_conf) + 1e-5)


# EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range.
backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(int(MODEL_H), int(MODEL_W), 3),
        pooling=None,
    )
yolo = Conv2D(512, 3, activation=None, padding="same")(backbone.layers[-1].output)
yolo = BatchNormalization()(yolo)
yolo = LeakyReLU()(yolo)
yolo = Conv2D(5, 1, activation=None, padding="same", name='YOLO')(yolo)
yolo = Activation(tf.keras.activations.linear)(yolo)
model = tf.keras.Model(inputs=backbone.inputs, outputs=yolo)

#opt = tf.keras.optimizers.SGD(momentum=0.9)
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss=my_loss, metrics=[recall, pre, ave_iou])

if not cfg.START_NEW_TRAINING:
    model.load_weights(cfg.MODEL_NAME)

model.optimizer.learning_rate.assign(cfg.LEARNING_RATE)

if cfg.PREDICT_MODE:
    DIV = cfg.DIV
    cut_tag(AUGM=True)
    images_to_numpy(random_flip=False, random_order=False, COLOR_AUG=False, BRIGHTNESS_AUG=False)
    xdata = np.load('xdata.npy')
    ydata = np.load('ydata.npy')
    for idx in range(xdata.shape[0]):                 # for each image
        preds = model(xdata[idx:idx+1,...], training=False).numpy()
        preds = preds[0,...]
        img = xdata[idx,...]/255
        for j in range(preds.shape[0]):        # h
            for k in range(preds.shape[1]):    # w                
                img[DIV//2+j*DIV, DIV//2+k*DIV, :] = [1,0,0]      # illustrate grid center
                conf = preds[j,k,0]                            # illustrate prediction
                if conf > 0.5:
                    box_xy = preds[j,k,1:3]                  # range 0-1
                    box_xy  = DIV * (2 * box_xy - [1.,1.])
                    grid_cen_xy = np.array([DIV//2+k*DIV, DIV//2+j*DIV])
                    box_xy  = grid_cen_xy + box_xy
                    box_wh = preds[j,k,3:5] * np.array([MODEL_W,MODEL_H]) * 0.25      # range 0-1
                    pt0 = box_xy - box_wh/2
                    pt1 = box_xy + box_wh/2
                    cv.rectangle(img, pt0.astype(int), pt1.astype(int), (0,0,1), 1)

                conf = ydata[idx,j,k,0]                             # illustrate ground true
                if conf > 0.5:
                    box_xy = ydata[idx,j,k,1:3]                  # range 0-1
                    box_xy  = DIV * (2 * box_xy - [1.,1.])
                    grid_cen_xy = np.array([DIV//2+k*DIV, DIV//2+j*DIV])
                    box_xy  = grid_cen_xy + box_xy
                    box_wh = ydata[idx,j,k,3:5] * np.array([MODEL_W,MODEL_H]) *0.25     # range 0-1
                    pt0 = box_xy - box_wh/2
                    pt1 = box_xy + box_wh/2                    
                    cv.ellipse(img, box_xy.astype(int), (box_wh/2).astype(int), 0, 0, 360, (0, 1, 1), 1)
                    
        cv.imwrite(os.path.join(cfg.OUTPUT_DIR,str(idx)+'.png'), img*255)
    print('Prediction completed')

else:
    fid = open('continue','w')
    fid.write('Delete this file to stop training.')
    fid.close()
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    batchsize = cfg.BATCH_SIZE
    gentr = Dynamo(batchsize)
    model.fit(gentr, 
        steps_per_epoch=gentr.xdata.shape[0]//batchsize, epochs=5000,
        callbacks=[CustomCallback(gentr), csv_logger])
