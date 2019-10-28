#!/usr/bin/env python
# coding: utf-8

# # Semantic Segmentation
# 
# *General structure following [fast.ai notebook on camvid](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb)*

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import PIL

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks.tracker import SaveModelCallback


# # Load Data
# 
# ### We use the [Berkely Deep Drive Dataset](https://bdd-data.berkeley.edu/) which contains a rich labeled dataset for image segmentation in diverse conditions (weather, city, reference car…).

# In[3]:


path_data = Path('/home/ruslan/Desktop/SelfDrivingCar/open_source/Semantic_Segmentation/semantic-segmentation/data/bdd100k/seg/')
path_lbl = path_data/'labels'
path_img = path_data/'images'


# ### Images and labels filenames

# In[4]:


fnames = get_image_files(path_img, recurse = True)

lbl_names = get_image_files(path_lbl, recurse = True)


# ### Now we need to create a function that maps from the path of an image to the path of its segmentation.

# In[7]:


get_y_fn = lambda x: path_lbl/x.parts[-2]/f'{x.stem}_train_id.png'


# ### We can now use the obtained label path to open a segmentation image.

# In[8]:


mask = open_mask(get_y_fn(img_f))
src_size = np.array(mask.shape[1:])

# size = src_size//4
# bs = 32
size = src_size
bs = 2


# In[11]:


# Classes extracted from dataset source code
# -> https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/label.py

segmentation_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
void_code = 19  # used to define accuracy and disconsider unlabeled pixels


# In[12]:


src = (SegmentationItemList.from_folder(path_img) # Load in x data from folder
       .split_by_folder(train='train', valid='val') # Split data into training and validation set 
       .label_from_func(get_y_fn, classes = segmentation_classes)) # Label data using the get_y_fn function


# ### Perform data augmentation and create databunch

# In[15]:


# Define transformations to apply 
import numpy as np

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


# ### We can show a few examples using the show_batch method which is available for all sorts of databunches

# It is also possible to create annotated segmenatation data from scratch by youe own, using such tools as:
# https://github.com/abreheret/PixelAnnotationTool

# ## Model creation and training

# A function that will measure the accuracy of the model. The accuracy in an image segmentation problem is the same as that in any classification problem.
# 
# Accuracy = no of correctly classified pixels / total pixels
# 
# However in this case, some pixels are labelled as Void and shouldn’t be considered when calculating the accuracy. Hence we make a new function for accuracy where we avoid those labels.

# In[17]:


def acc(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[18]:


iou = partial(dice, iou=True)
f_score = partial(fbeta, thresh=0.2)
# metrics=[acc, iou, f_score]
metrics=[acc]

wd=1e-5 # weight decay

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd).to_fp16()


# With our model ready to go we can now search for a fitting learning rate and then start training our model
lr_find(learn)


# To read about picking a learning rate, go to:
# https://towardsdatascience.com/fastai-image-classification-32d626da20
# We need to select a point on the graph with the fastest decrease in the loss.

lr=3e-3 # pick a learning rate

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

# learn.fit_one_cycle(10, slice(lr),
#                     callbacks=[SaveModelCallback(learn, name='best_model', every='epoch', monitor='accuracy')])#, pct_start=0.9) # train model


# Standardly only the decoder is unfrozen, which means that our pretrained encoder didn’t receive any training yet so we will now show some results and then train the whole model.

# In[ ]:


import os
current_path = os.getcwd()
# learn.save(current_path + '/trained_models/berkeley-stage-1') # save model
learn.save('berkeley-full-size')
learn.show_results(rows=3, figsize=(20,10))


# ### Perform fine-tuning of all layers

# In[ ]:


# learn.load(current_path + '/trained_models/berkeley-stage-1');
learn.load('berkeley-full-size');

learn.unfreeze() # unfreeze all layers

lrs = slice(lr/400,lr/4)

learn.fit_one_cycle(12, lrs)

learn.save('berkeley-full-size')

# ### Video inference

# In[ ]:


def np_img2fastai_img(np_img):
    img_fastai = Image(pil2tensor(np_img, dtype=np.float32).div_(255))
    return img_fastai

import imutils
import cv2
from tqdm import tqdm

vs = cv2.VideoCapture('../data/videos/moscow_streets.mp4')
writer = None

# try to determine the total number of frames in the video file
try:
    prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2()         else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    total = -1

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(segmentation_classes) - 1, 3), dtype="uint8")
COLORS = np.vstack([COLORS, [0, 0, 0]]).astype("uint8")

while(1):
# for i in tqdm( range(30) ):
    # get raw frames from video stream
    ret, frame = vs.read()
    if ret == False:
        break
    # preprocess raw frames
    start = time.time()
    frame_fastai = np_img2fastai_img(frame)
    output = learn.predict( frame_fastai )
    end = time.time()
#     print("[INFO] single frame took {:.4f} seconds".format(end-start))
    
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('../data/videos/moscow_output_berkeley.avi', fourcc, 30,
            (size[1], size[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time: {:.4f} [min]".format(elap * total/60.))

    # preprocess output frames before writing to disk
    classMap = np.array(output[1][0], dtype=np.uint8)
    mask = COLORS[classMap]
    frame_resized = cv2.resize(frame, (size[1], size[0]), interpolation = cv2.INTER_AREA)
    output_to_write = (0.3 * frame_resized + 0.7 * mask).astype("uint8")
    # write the output frame to disk
    writer.write(output_to_write)

