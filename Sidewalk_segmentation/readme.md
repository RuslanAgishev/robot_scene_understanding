## Sidewalk segmentation
Main road segmentation in street environment for mobile robots.

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/figures/sidewalk_segmentation.png"/>

The main goal of this approach is labeling every pixel on input image which corresponds to a drivable area.

Convolutional Neuron Networks (CNNs) architectures considered here:
- [Unet](https://arxiv.org/abs/1505.04597) based on [keras implementation](https://github.com/qubvel/segmentation_models)

Training pipeline is available [here](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Sidewalk_segmentation/train_sidewalk_segmentation.ipynb).
For models inference example take a look at this [code](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Sidewalk_segmentation/inference_keras_berkeley_sidewalk.ipynb)

### Results (NVIDIA Jetson Nano)
| Image Size | Dataset                     | Backbone             | FPS  | F1-score |IoU   | Model    |
|:----------:|:---------------------------:|:--------------------:|:----:|:--------:|:----:|:--------:|
| (192, 320) | Berkeley-DD (Drivable Area) | Unet-ResNet18        | 4.9  | 86.9     | 79.8 |[unet-resnet18](https://drive.google.com/open?id=1QBXYaWzorfyhekkL2IkYRKbz6WubNIU2) |
| (192, 320) | Berkeley-DD (Drivable Area) | Unet-Mobilenetv2     | 5.1  | 86.0     | 78.9 |[unet-mobilenetv2](https://drive.google.com/open?id=1SSWdTMOH-wHtyOnOJELKN0c4cTRRFxtN) |
