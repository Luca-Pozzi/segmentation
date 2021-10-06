# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog #DatasetCatalog

# Path declarations
MODELS_DIR = "/root/tiago_public_ws/src/segmentation/models"
CONFIG_FILE_DIR = "COCO-InstanceSegmentation/"
# Default values declarations
MODEL_NAME = "mask_rcnn_R_50_FPN_3x"
DISPLAY_TIME = 5000 #[ms]
SCORE_THRESHOLD = 0.75

def predictor_configuration(model_name = MODEL_NAME, score_threshold = SCORE_THRESHOLD):
	##########
	# This function initializes the predictor by choosing the model and setting some parameters. This initialization is
	# kept separated from the actual segmentation to make it faster when segment_image is put into a loop.
	# model_name:		the .pkl file with the network weights. It must be placed in MODELS_DIR and must be one of the 
	#					pre-trained model available in Detectron2 model-zoo.
	# score_threshold:	only the instances with a score higher than this value are kept.
	##########

	# Create a detectron2 config and detectron2 DefaultPredictor
	cfg = get_cfg() 
	# set cfg by looking at the proper config file in the 'model_zoo' folder of the detectron2 library
	# if the selected model does not have the corresponding .yaml file in the model_zoo, a default CNN is used instead
	try:
		model_zoo.get_config_file(CONFIG_FILE_DIR + model_name + ".yaml")
	except RunTimeError:
		model_name = MODEL_NAME
		print("The .yaml configuration file corresponding to the specified model has not been found.\nThe default " +
		 MODEL_NAME[:-4] + "has been used instead.")

	cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_DIR + model_name + ".yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

	# Load the weights of the selected model
	cfg.MODEL.WEIGHTS = MODELS_DIR + "/" + model_name + ".pkl"
	predictor = DefaultPredictor(cfg)	# initialize a predictor with the loaded weights
	return cfg, predictor

def segment_image(img, cfg, predictor, visualize = False):
	############
	# img: 				the image to be segmented. File format tested so on: .jpg
	# cfg: 				the configuration of the predictor (including model weights and class number/label correspondence)
	# predictor:		the actual predictor object
	# visualize: 		if True the image with the superimposed annotation is concatenated to the original image
	#					and displayed. Leave it False (as by default) to use the function as a segmentation tool only.
	############
	outputs = predictor(img)
	
	# add the annotations to the original image
	v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)	# not needed to import the whole cfg, MetadataCatalog.get(cfg.DATASET.TRAIN[0]) would be enough		
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	original_img = img
	annotated_img = out.get_image()[:, :, ::-1]
	'''
	if visualize:	# this part is skipped when using these functions inside a ROS node to leave the visualization 
					# part to a dedicated node (if needed)
		img_concatenate = np.concatenate((original_img, annotated_img),axis = 1)
		cv2.imshow("segmentation result", img_concatenate)
		cv2.moveWindow("segmentation result",180,180)
		cv2.waitKey(DISPLAY_TIME)
		cv2.destroyAllWindows()
	'''
	return annotated_img, outputs

if __name__ == '__main__':
	print("This script is supposed to provide useful functions and not to be run alone.\nAnyway, here is a sample.")
	default_img = cv2.imread("sample_image.jpg")
	cfg = predictor_configuration()
	segment_image(default_img, cfg, visualize = True)

	