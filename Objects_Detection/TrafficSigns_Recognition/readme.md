## Traffic signs recognition
GTSRB (German Traffic Sign Recognition Benchmark) dataset can be downloaded from
[kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/)

The project is built based on PyImageSearch [tutorial](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/).

A pre-trained keras model is located [here](https://github.com/RuslanAgishev/robot_scene_understanding/tree/master/Objects_Detection/TrafficSigns_Recognition/output/trafficsignnet.model).
The whole data preparation, training and testing pipeline is available [here](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Objects_Detection/TrafficSigns_Recognition/keras_pipeline.ipynb).

### Training
Note, that you need to download the GTSRB dataset and put it in the current folder
for the next commands to work.

In order to train the model execute:
```bash
python3 train.py --dataset gtsrb-german-traffic-sign \
                 --model output/trafficsignnet.model --plot output/plot.png
```
or simply: ```./train.sh```

### Testing
For inference on a sequence of images:
```bash
python3 predict.py --model output/trafficsignnet.model \
	           --images gtsrb-german-traffic-sign/Test --examples examples
```
or simply: ```./predict.sh```

### Results
| Dataset | Backbone             | Precision | Recall | F1-score | FPS                              | Model    |
|:-------:|:--------------------:|:---------:|:------:|:--------:|:--------------------------------:|:--------:|
| GTSRB   | [TrafficSignNet](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Objects_Detection/TrafficSigns_Recognition/pyimagesearch/trafficsignnet.py)             | 0.91      | 0.88   | 0.88     | 77.9 (GeForce RTX 2080), 10.5 (Intel Core i5-6200U CPU) |[weights](https://github.com/RuslanAgishev/robot_scene_understanding/blob/master/Objects_Detection/TrafficSigns_Recognition/output/trafficsignnet1572968296.model)
