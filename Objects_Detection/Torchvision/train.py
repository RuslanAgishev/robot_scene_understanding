#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import torch
import json
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
from torch import Tensor
import utils
from PIL import Image
import transforms as T
from tqdm import tqdm
from torch import nn

# Helper functions
from engine import train_one_epoch, evaluate
import utils
import transforms as T

def _load_json(path_list_idx):
    with open(path_list_idx, 'r') as file:
        data = json.load(file)
    return data

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# Note that we do not need to add a mean/std normalization
# nor image rescaling in the data transforms, as those are
# handled internally by the Mask R-CNN model.

def get_ground_truths(train_img_path_list,anno_data):
    bboxes,total_bboxes = [],[]
    labels,total_labels = [],[]
    classes =  {'bus':0, 'traffic light':1, 'traffic sign':2, 'person':3, 'bike':4, 'truck':5, 'motor':6, 'car':7, 
            'train':8,'rider':9,'drivable area':10,'lane':11}

    for i in tqdm(range(len(train_img_path_list))):
        for j in range(len(anno_data[i]['labels'])):
            if 'box2d' in anno_data[i]['labels'][j]:
                xmin = anno_data[i]['labels'][j]['box2d']['x1']
                ymin = anno_data[i]['labels'][j]['box2d']['y1']
                xmax = anno_data[i]['labels'][j]['box2d']['x2']
                ymax = anno_data[i]['labels'][j]['box2d']['y2']
                bbox = [xmin,ymin,xmax,ymax]
                category = anno_data[i]['labels'][j]['category']
                cls = classes[category]

                bboxes.append(bbox)
                labels.append(cls)

        total_bboxes.append(Tensor(bboxes))
        total_labels.append(Tensor(labels))
        bboxes=[]
        labels=[]

    return total_bboxes,total_labels


# Defining the Dataset
# 
# The [torchvision reference scripts for training object detection, instance segmentation and person keypoint detection](https://github.com/pytorch/vision/tree/v0.3.0/references/detection) allows for easily supporting adding new custom datasets.
# The dataset should inherit from the standard `torch.utils.data.Dataset` class, and implement `__len__` and `__getitem__`.
# 
# The only specificity that we require is that the dataset `__getitem__` should return:
# 
# * image: a PIL Image of size (H, W)
# * target: a dict containing the following fields
#     * `boxes` (`FloatTensor[N, 4]`): the coordinates of the `N` bounding boxes in `[x0, y0, x1, y1]` format, ranging from `0` to `W` and `0` to `H`
#     * `labels` (`Int64Tensor[N]`): the label for each bounding box
#     * `image_id` (`Int64Tensor[1]`): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
#     * `area` (`Tensor[N]`): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
#     * `iscrowd` (`UInt8Tensor[N]`): instances with `iscrowd=True` will be ignored during evaluation.
#     * (optionally) `masks` (`UInt8Tensor[N, H, W]`): The segmentation masks for each one of the objects
#     * (optionally) `keypoints` (`FloatTensor[N, K, 3]`): For each one of the `N` objects, it contains the `K` keypoints in `[x, y, visibility]` format, defining the object. `visibility=0` means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt `references/detection/transforms.py` for your new keypoint representation
# 
# If your model returns the above methods, they will make it work for both training and evaluation, and will use the evaluation scripts from pycocotools.
# 
# Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratio), then it is recommended to also implement a `get_height_and_width` method, which returns the height and the width of the image. If this method is not provided, we query all elements of the dataset via `__getitem__` , which loads the image in memory and is slower than if a custom method is provided.
# 

# In[4]:


bdd_path = '/home/ruslan/Desktop/Berkeley_DeepDrive/bdd100k/'

root_img_path = os.path.join(bdd_path, 'images','100k')
root_anno_path = os.path.join(bdd_path, 'labels')

train_img_path = root_img_path+'/train/'
val_img_path = root_img_path+'/val/'

train_anno_json_path = root_anno_path+'/bdd100k_labels_images_train.json'
val_anno_json_path = root_anno_path+'/bdd100k_labels_images_val.json'

with open("datalists/bdd100k_train_images_path.txt", "rb") as fp:
    train_img_path_list = pickle.load(fp)
with open("datalists/bdd100k_val_images_path.txt", "rb") as fp:
    val_img_path_list = pickle.load(fp)


class BerkeleyDD(torch.utils.data.Dataset):
    def __init__(self, img_path, anno_json_path, transforms=None):
        super(BerkeleyDD, self).__init__()
        self.img_path = img_path
        self.anno_data = _load_json(anno_json_path)
        self.total_bboxes_list,self.total_labels_list = get_ground_truths(self.img_path, self.anno_data)
        self.transforms = transforms
        self.classes =  {'bus':0, 'traffic light':1, 'traffic sign':2,
                         'person':3, 'bike':4, 'truck':5, 'motor':6,
                         'car':7, 'train':8, 'rider':9,
                         'drivable area':10,'lane':11}

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self,idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert("RGB")
        
        labels = self.total_labels_list[idx]
        bboxes = self.total_bboxes_list[idx]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        
        img_id = Tensor([idx])
        iscrowd = torch.zeros(len(bboxes,), dtype=torch.int64)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels.type(torch.int64)
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

print("Loading files")
batch_size = 4

dataset_train = BerkeleyDD(train_img_path_list[:100], train_anno_json_path, get_transform(train=True))
dataloader_train =  torch.utils.data.DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=utils.collate_fn)

dataset_val = BerkeleyDD(val_img_path_list[:10], val_anno_json_path, get_transform(train=False))
dataloader_val =  torch.utils.data.DataLoader(dataset_val,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=utils.collate_fn)


# Defining your model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detection_model(num_classes):
    # load an detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one
    return model.cuda()


num_epochs = 10
lr = 1e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Model initialization")
model = get_detection_model(len(dataset_train.classes))
# move model to the right device
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3,max_lr=6e-3)

try:
    os.mkdir('saved_models/')
except:
    pass


# And now let's train the model for 10 epochs, evaluating at the end of every epoch.
print('Training started')

for epoch in tqdm(range(num_epochs)):
    train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=200)
    lr_scheduler.step()
    
    #if epoch==5 or epoch==10 or epoch==15 or epoch == 20 or epoch==24:
    save_name = 'saved_models/bdd100k_' + str(epoch)+'.pth'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_name)
    print("Saved model", save_name)


