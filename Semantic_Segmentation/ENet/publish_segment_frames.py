# USAGE
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/massachusetts.mp4 --output output/massachusetts_output.avi
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4 --output output/toronto_output.avi

# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf

# construct the argument parse and parse the arguments
model = "enet-cityscapes/enet-model.net"
classes = "enet-cityscapes/enet-classes.txt"
video = "videos/moscow_streats.mp4"
colors = "enet-cityscapes/enet-colors.txt"
width = 500 # desired width (in pixels) of input image

# load the class label names
CLASSES = open(classes).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if colors:
	COLORS = open(colors).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class
# label
else:
	# initialize a list of colors to represent each class label in
	# the mask (starting with 'black' for the background/unlabeled
	# regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(model)

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(video)

# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
rospy.init_node("segmenter", anonymous=True)
pub = rospy.Publisher('segmented_images', Image, queue_size=10)
br = CvBridge()
rate = rospy.Rate(10)
while not rospy.is_shutdown():
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# construct a blob from the frame and perform a forward pass
	# using the segmentation model
	frame = imutils.resize(frame, width=width)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
        output = net.forward()
	end = time.time()
	print(end-start)

	# infer the total number of classes along with the spatial
	# dimensions of the mask image via the shape of the output array
	(numClasses, height, width) = output.shape[1:4]

	# our output class ID map will be num_classes x height x width in
	# size, so we take the argmax to find the class label with the
	# largest probability for each and every (x, y)-coordinate in the
	# image
	classMap = np.argmax(output[0], axis=0)

	# given the class ID map, we can map each of the class IDs to its
	# corresponding color
	mask = COLORS[classMap]

	# resize the mask such that its dimensions match the original size
	# of the input frame
	mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
		interpolation=cv2.INTER_NEAREST)

	# perform a weighted combination of the input frame with the mask
	# to form an output visualization
	output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")
        pub.publish(br.cv2_to_imgmsg(output))
        rate.sleep()

# release the file pointers
print("[INFO] cleaning up...")
vs.release()
