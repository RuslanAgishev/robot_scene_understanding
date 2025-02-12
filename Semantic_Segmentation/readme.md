## Semantic Segmentation
Road scene semantic segmentation for autonomous vehicles.

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/ENet/output/munich_seg_output.png" width="400" /> <img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/ENet/output/example_02_seg_output.png" width="400" />

The main goal of this approach is labeling every pixel on input image with ID
corresponding to a separate class of road scene objects.
Convolutional Neuron Networks (CNNs) architectures considered here:
- [Enet](https://arxiv.org/abs/1606.02147) based on [PyImageSearch](https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/) tutorial
- [Unet](https://arxiv.org/abs/1505.04597) based on [keras implementation](https://github.com/qubvel/segmentation_models) and [fastai-pytorch tutorial](https://course.fast.ai/videos/?lesson=3)

### Results (NVIDIA GeForce RTX 2080)
| Image Size | Dataset     | Backbone             | FPS  | Accuracy |IoU   | F1-score | Pipeline | Model
|:----------:|:-----------:|:--------------------:|:----:|:--------:|:----:|:--------:|:--------:|:----------:|
| (360, 480) | CamVid      | Unet-ResNet34        | 23.8 | 89.9     | 41.8 (32 classes) |          | [jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/UNet/fastai/fastai_camvid.ipynb) | [camvid-stage-2](https://drive.google.com/open?id=1Dn8KDbPyB4WRhh-a-stBeqq8fWxvSlKt)
| (384, 640) | CamVid      | Unet-EfficientNetB3  | 28.3 |          | 65.2 (8 classes) | 74.3     | [jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/UNet/keras/multiclass_segmentation_camvid.ipynb) | [unet-camvid](https://drive.google.com/open?id=1KKQVrwir0yJ_dCga2EWW9F23uBuO4V4t)
| (512, 1024)| CityScapes  | Unet-EfficientNetB3  | 14.9 |          | 63.8 (8 classes) | 73.4     | [jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/UNet/keras/multiclass_segmentation_cityscapes.ipynb) | [unet-cityscapes](https://drive.google.com/open?id=1zK_GZsLIHcD7X-cDk7Z6VajZlBmMPxHy)
| (720, 1280)| Berkeley-DD | Unet-ResNet34        | 4.3  | 87.7     | 45.3 (20 classes) |          | [jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/UNet/fastai/fastai_berkeley.ipynb) | [berkeley-full-size-2](https://drive.google.com/open?id=1fVNFaNOtSm7QvknIokIsIgzyGahltxhr)
| (352, 640) | Berkeley-DD | Unet-EfficientNetB3  | 26.8 |          | 56.9 (20 classes) | 61.1     | [jupyter](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Semantic_Segmentation/UNet/keras/multiclass_segmentation_berkeley.ipynb) | [unet-berkeley](https://drive.google.com/open?id=19Hv_3je6Xt8-NzW4RoroTJGxj2uRFNdC)

Semantic segmentation video results are available on [Google-drive](https://drive.google.com/open?id=1rH6Zj5QBcv2eQGxh8kpIKQEz7AhU1bHu).
