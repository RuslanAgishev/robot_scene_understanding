# Semantic Segmentation based on ENet CNN architecture
Pre-trained on [CityScapes](http://cityscapes-dataset.com/) dataset ENet model is stored in enet-cityscapes/ folder.
In order to train ENet model from scratch based on Caffe framework follow the next [tutorial](https://github.com/TimoSaemann/ENet/tree/master/Tutorial)
provided by ENet authors.

To test the pretrained model on an image execute:
```bash
./run_img_segment.sh
```
or:
```bash
python segment.py --model enet-cityscapes/enet-model.net\
		  --classes enet-cityscapes/enet-classes.txt \
		  --colors enet-cityscapes/enet-colors.txt\
		  --image images/example_01.png
```
The following command will run a sequence of ENet inferences on frames from a video file:
```bash
./run_video_segment.sh
```
or:
```bash
python segment_video.py --model enet-cityscapes/enet-model.net\
			--classes enet-cityscapes/enet-classes.txt\
			--colors enet-cityscapes/enet-colors.txt\
			--video videos/massachusetts.mp4\
			--output output/massachusetts_output.avi\
			--show 1
```
