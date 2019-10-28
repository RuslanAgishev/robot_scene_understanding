## CNN-based object detection based on [YOLO](https://arxiv.org/abs/1506.02640)-algorithm
### Usage:
- Download pretrained model architecture and weighs:

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

For more information take a look at the [PyImageSearch tutorial](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/).
