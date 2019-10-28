## CNN-based object detection based on [YOLO](https://arxiv.org/abs/1506.02640)-algorithm

<img src="https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Objects_Detection/YOLO/output/example_04_output.png" width="800"/>

### Usage:
- Download pretrained model architecture and weights:
```bash
./get_model.sh
```
- Try YOLO-object detection on images:
```bash
./image_inference.sh
```
or:
```bash
python yolo.py --image images/example_01.png --yolo yolo-coco
```

- Apply pretrained model to a video. The output will be saved in the output/ folder:
```bash
./video_inference.sh
```
or:
```bash
python yolo_video.py --input videos/car_chase_01.mp4 --output output/car_chase_01_output.avi --yolo yolo-coco
```

In order to train YOLO-model refer to the [darknet documentation](https://pjreddie.com/darknet/yolo/).

For more information take a look at the [PyImageSearch tutorial](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/).
