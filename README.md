# Train Tracks segmentation and Switches classification
By Andrei-Robert Alexandrescu, 931
(Dataset: RailSem19 https://wilddash.cc/railsem19)

## Contents
### Files

- seg: contains the segmentation files
	- dataset.py: dataloader for semantic segmentation. An item is a pair (image,mask).
	- model.py: UNet model implementation
	- model_resunet.py: ResUNet model implementation
	- model_resunetpp.py: ResUNet++ model implementation
	- train.py: where the training takes place
	- video_inference.py: for creating demo videos
	- utils.py

- cla: contains the classification files
	- bounding_boxes.py: script for extracting only the desired bounding boxes from the raw dataset
	- image_generator.py: takes the images from the bounding_boxes.py script,
	selects the bounding boxes and creates a dataset based on three hyperparameters
	alpha, beta, gamma discussed in the paper in the dataset subsection of Methodology.
	- dataset.py: dataloader for switches. An item is a pair (image,class)
	- main.py: where the training takes place
	- utils.py
	- models:
		- vgg.py: allows for multiple sizes (11, 13, 16, 19)
		- resnet.py
	