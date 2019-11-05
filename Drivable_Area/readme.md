## Drivable Area Segmentation
Lane lines segmentation for autonomous vehicles.

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/figures/drivable_area.png"/>

The main goal of this approach is labeling of pixels on input image which belong to road lane lines.

Convolutional Neuron Networks (CNNs) architectures considered here:
- [Unet](https://arxiv.org/abs/1505.04597) based on [fastai-pytorch tutorial](https://course.fast.ai/videos/?lesson=3)

### Results (NVIDIA GeForce RTX 2080)
| Image Size | Dataset                     | Backbone             | FPS  | Accuracy |IoU   | Model    |Pipeline |
|:----------:|:---------------------------:|:--------------------:|:----:|:--------:|:----:|:--------:|:-------:|
| (360, 640) | Berkeley-DD (Drivable Area) | Unet-ResNet34        | 25.9 |          |      |[unet-drivable](https://drive.google.com/open?id=1fCEsEvzFMLxTxtw4EIdiQWonyXW8bOlf) |[jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Drivable_Area/fastai_berkeley.ipynb)
