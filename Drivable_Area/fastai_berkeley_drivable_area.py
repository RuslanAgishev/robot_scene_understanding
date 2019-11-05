#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import PIL
from tqdm import tqdm
import numpy as np

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks.tracker import SaveModelCallback


# Load Data
path_data = Path('/home/ruslan/Desktop/Berkeley_DeepDrive/bdd100k/')
path_lbl = path_data/'drivable_maps'/'labels'
path_img = path_data/'images'/'100k'


# ### Images and labels filenames
fnames = get_image_files(path_img, recurse = True)
lbl_names = get_image_files(path_lbl, recurse = True)

img_f = fnames[100]

# Now we need to create a function that maps from the path of an image to the path of its segmentation.
get_y_fn = lambda x: path_lbl/x.parts[-2]/f'{x.stem}_drivable_id.png'

# We can now use the obtained label path to open a segmentation image.
mask = open_mask(get_y_fn(img_f))

src_size = np.array(mask.shape[1:])


# Datasets
# Now that we know how our data looks like we can create our data-set using the SegmentationItemList class provided by FastAI.
# size = src_size//4
# bs = 32
size = src_size//2
bs = 8

# Classes extracted from dataset source code
# -> https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py
segmentation_classes = [
    'background', 'current', 'neighbour'
]

src = (SegmentationItemList.from_folder(path_img) # Load in x data from folder
       .split_by_folder(train='train', valid='val') # Split data into training and validation set
       .label_from_func(get_y_fn, classes = segmentation_classes) # Label data using the get_y_fn function
       )

# ### Perform data augmentation and create databunch

# Define transformations to apply 

train_tfms = [
        brightness(change=(0.1, 0.9), p=0.5),
        rotate(degrees=(-20,20), p=0.5),
        contrast(scale=(0.5, 2.), p=0.5),
        jitter(magnitude=np.random.uniform(-0.05, 0.05), p=0.1),
        symmetric_warp(magnitude=(-0.2,0.2), p=0.5),
        zoom(scale=np.random.uniform(1,1.6), p=0.5)
]

valid_tfms = []
# valid_tfms = train_tfms

# transformations = [train_tfms, valid_tfms]
transformations = get_transforms()

data = (src.transform(transformations, size=size, tfm_y=True)
        .databunch(bs=bs) # Create a databunch
        .normalize(imagenet_stats)) # Normalize for resnet

# Model creation and training

# A function that will measure the accuracy of the model. The accuracy in an image segmentation problem is the same as that in any classification problem.
# Accuracy = no of correctly classified pixels / total pixels 
# However in this case, some pixels are labelled as Void and shouldn’t be considered when calculating the accuracy. Hence we make a new function for accuracy where we avoid those labels.
def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

iou = partial(dice, iou=True)
f_score = partial(fbeta, thresh=0.2)
# metrics=[acc, iou, f_score]
metrics=[]

wd=1e-6 # weight decay


# To create a U-NET in FastAI the unet_learner class can be used. We not only going to pass it our data but we will also specify an encoder-network (Resnet34 in our case), our accuracy function as well as a weight-decay
#learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd).to_fp16()
learn = unet_learner(data, models.resnet34, wd=wd).to_fp16()
# To read about picking a learning rate, go to:
# https://towardsdatascience.com/fastai-image-classification-32d626da20
# We need to select a point on the graph with the fastest decrease in the loss.
lr=5e-5 # pick a learning rate

learn.fit_one_cycle(20, slice(lr),
                    callbacks=[SaveModelCallback(learn, name='best_model',
                    							 every='epoch',
                    							 monitor='accuracy')])# train model

learn.save('unet-resnet34-drivable-area-berkeley')
print('Done!')
