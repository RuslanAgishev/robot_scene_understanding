## Object Detection
Traffic scene object detection based on [Berkeley DeepDrive](https://bdd-data.berkeley.edu/) dataset.

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/figures/objects_detection2.png"/>
<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/figures/objects_detection1.png"/>

[Faster-RCNN](https://arxiv.org/abs/1506.01497) architecture is utilized for objects detection based on torchvision
[tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
For the general data preparation, training and testing pipeline please refer to the following
[jupyter-notebook](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Objects_Detection/Torchvision/Faster_R-CNN_full_pipeline.ipynb)

### Results (NVIDIA GeForce RTX 2080)
| Dataset                        | Backbone             | FPS  | Model    |
|:------------------------------:|:--------------------:|:----:|:--------:|
| Berkeley-DD (Object Detection) | Faster-RCNN          | 15.7 |[faster-rcnn](https://drive.google.com/open?id=1MbPiDjvhA0N_o7DgqGt9pFJ7m6UIFxOk)

Objects detection video results are available on [Google-drive](https://drive.google.com/drive/folders/1PUdsOCB84b_j0w5T53ukSAZW3NzvHLr5).
