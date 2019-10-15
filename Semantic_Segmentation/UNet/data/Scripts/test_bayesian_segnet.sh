 
python ../Scripts/test_bayesian_segnet.py \
	--model ../Models/bayesian_segnet_inference.prototxt \
	#--weights ../Models/Inference/test_weights.caffemodel \
	--colours ../Scripts/camvid11.png \
	--data ../CamVid/test.txt
