import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
keras.backend.set_session(tf.Session(config=config))

import segmentation_models as sm
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

print('[INFO] Is GPU available: {}'.format(tf.test.is_gpu_available()))

BACKBONE = 'resnet18'
CLASSES = ['current']

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = len(CLASSES)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (background: 0.5; current: 2.; neighbour: 2.; )
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 2, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# compile keras model with defined optimozer, loss and metrics
model.compile('Adam', total_loss)

# load best weights
model.load_weights('/home/tegraboy/Desktop/best_model.h5')

preprocess_input = sm.get_preprocessing(BACKBONE)
def image_preprocessing(img_path, size=(192, 320)):
    image = cv2.imread(img_path)
    # convert color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply preprocessing
    image = cv2.resize(image, (size[1], size[0]), cv2.INTER_LINEAR)
    image = preprocess_input(image)
    img_processed = np.expand_dims(image, axis=0)
    return img_processed

import os
PATH = '/home/tegraboy/Desktop/example_images/'
for img in os.listdir(PATH):
    input_image = image_preprocessing(os.path.join(PATH, img))
    # predict segmentation mask from the frame
    start = time.time()
    pr_layers = model.predict(input_image)
    dt = time.time() - start
    print("[INFO] single frame of shape {} took {:.4f} seconds".format(input_image.shape[1:3], dt))

