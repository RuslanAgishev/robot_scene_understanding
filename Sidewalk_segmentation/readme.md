## Sidewalk segmentation
Main road segmentation in street environment for mobile robots.

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/tree/master/figures/sidewalk_segmentation.png" />

The main goal of this approach is labeling every pixel on input image which corresponds to a drivable area.

Convolutional Neuron Networks (CNNs) architectures considered here:
- [Unet](https://arxiv.org/abs/1505.04597) based on [keras implementation](https://github.com/qubvel/segmentation_models)

### Results (NVIDIA Jetson Nano)
| Image Size | Dataset                     | Backbone             | FPS  | F1-score |IoU   | Model    |
|:----------:|:---------------------------:|:--------------------:|:----:|:--------:|:----:|:--------:|
| (192, 320) | Berkeley-DD (Drivable Area) | Unet-ResNet18        |      |          |      |[unet-resnet18]() |
| (192, 320) | Berkeley-DD (Drivable Area) | Unet-Mobilenetv2     | 5.1  |          |      |[unet-mobilenetv2]() |

