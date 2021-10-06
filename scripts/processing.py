#! /root/venv/cv/bin/python
# This node is an adaptation of the feature_detector script.
# The scripts defines a subscriber and a publisher node. The subscriber receives an image, segments it 
# (exploiting the detectron2 library) and publishes the results of the segmentation as a custom message
# containing the annotated_img and the mask and bounding box of the instances belonging to meaningful 
# (user-defined) classes.

# Python libs
import sys, time
import numpy as np
import cv2
# ROS libraries
import rospy
# ROS messages
from sensor_msgs.msg import CompressedImage
from segmentation.msg import SegmentationOutcome    # custom messages, see the msg folder for further
                                                    # info about fields and data types 
# my scripts
from img_segmentation import predictor_configuration, segment_image

# constants and params
VERBOSE = False
CNN_MODEL_NAME = "mask_rcnn_R_50_FPN_3x"
SELECTED_CLASSES = [39] #, 40, 41, 60]
SELECTED_CLASS_NAMES = ['bottle'] #, 'wine glass', 'cup', 'dining table']
# only the mask of the instances belonging to the SELECTED_CLASSES are published. Set this to [0:80] to publish 
# all the masks.
# NOTE: the annotated_img includes annotations for the images belonging to ALL the classes, not only the once in
# specified in SELECTED_CLASSES.
# NOTE: the order of the elements in SELECTED_CLASSES and in SELECTED_CLASS_NAMES is used to find corresponding
# element. Using a dictionary or an external file would be better.


class ImageProcessing:
    def __init__(self):
        ### Initialize ROS publisher and ROS subscriber
        # topic where we publish
        self.segmentation_pub = rospy.Publisher("/segmentation_result", SegmentationOutcome, queue_size = 1)
        self.annotated_img_pub = rospy.Publisher("/segmentation_result/annotated_image/compressed", CompressedImage, queue_size = 1)
        # subscribed topic
        self.subscriber = rospy.Subscriber("/rgb_image", CompressedImage, self.segment,  queue_size = 1)
        ##########
        # Insert here the initialization of the feature detector/
        # segmentation algorithm.
        self.predictor_cfg, self.predictor = predictor_configuration(model_name = CNN_MODEL_NAME, 
            score_threshold = rospy.get_param("/detection_score_threshold"))
        #########
        if VERBOSE :
                print("Subscribed to RGB image topic, remapped to /rgb_image.")

    def format_classes(self, pred_classes_list, selected_classes_list = SELECTED_CLASSES, selected_class_names_list = SELECTED_CLASS_NAMES):
        # add to msg.classes all the elements of pred_classes_list that are present in selected_classes_list
        sel_classes_set = set(selected_classes_list)
        self.msg.classes = [selected_class_names_list[selected_classes_list.index(pred_class)] for pred_class in pred_classes_list if pred_class in sel_classes_set]

    def format_masks(self, pred_masks_list, pred_classes_list, selected_classes_list = SELECTED_CLASSES):
        # initialize an empty masks list of proper length
        self.msg.masks = [None] * len(self.msg.classes)
        output_list_idx = 0 # initialize the index specifying the position of each mask in the published list
        for idx in range(len(pred_masks_list)):
            if pred_classes_list[idx] in selected_classes_list:     # only the masks of the instances belonging to the
                                                                    # selected_classes_list are published
                # get the boolean(i.e. original) mask
                bool_mask = pred_masks_list[idx]
                # get the size of the image
                height = len(bool_mask)
                width = len(bool_mask[0])
                # convert the boolean mask into a CompressedImage 
                # (sub-optimal solution wrt publishing the boolean mask directly)
                numpy_mask = np.zeros((height, width))
                numpy_mask[bool_mask] = 255
                # append the mask to the list of masks that is going to be published
                self.msg.masks[output_list_idx] = CompressedImage()
                self.msg.masks[output_list_idx].data = cv2.imencode('.jpg', numpy_mask)[1].tostring()
                # format the header
                self.msg.masks[output_list_idx].header.stamp = rospy.Time.now()   
                # increment the index
                output_list_idx += 1

    '''
    def format_bbox(self, pred_boxes, pred_classes_list, selected_classes_list = SELECTED_CLASSES):
        # flatten the bbox list
        pred_boxes_list = []
        for idx in range(len(pred_classes_list)):
            if pred_classes_list[idx] in selected_classes_list: # only the boxes of the instances belonging to the
                                                                # selected_classes_list are published
                for vertex in pred_boxes[idx]:
                    pred_boxes_list.append(vertex.tolist())
        #pred_boxes_list = [vertex.tolist() if pred_classes_list for bbox in pred_boxes for vertex in bbox]
        # the bouding boxes are going to be reconstructed by the subscriber (relying on the fact that every four elements
        # in the list, a box is defined)
        if len(pred_boxes_list) % 4 == 0:
            # if everything seems to be ok, add the list to the message that is going to be published
            self.msg.bboxes = pred_boxes_list
        else:
            # if the above condition is not met, thrown an error message and do not publish
            print('ERROR: the number of vertexes of the bounding boxes is not a multiple of 4.')
	'''

    def segment(self, ros_data):
        # Callback function of subscribed topic (could be "camera" or "coco"). 
        # Here images get converted and segmented
        if VERBOSE :
            print('Received image of type: ' + ros_data.format)

        # Conversion to cv2 image
        np_arr = np.fromstring(ros_data.data, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        ##########
        # Insert here the feature detection/segmentation section.
        # The input and output of this section must be cv2 images.
        # It would be ideal to keep this section on a separate script.
        annotated_img_np, outputs = segment_image(img_np, self.predictor_cfg, self.predictor)
        ###########

        # Add the annotated image in the format of a CompressedImage to the custom message that is going to be published
        self.msg = SegmentationOutcome()    # custom message
        '''
        # header and format
        self.msg.annotated_img.header.stamp = rospy.Time.now()
        self.msg.annotated_img.format = "jpeg"
        # actual image
        self.msg.annotated_img.data = np.array(cv2.imencode('.jpg', annotated_img_np)[1]).tostring()
        '''
        # Create a custom message with the predicted classes and the corresponding bounding boxes and masks
        self.format_classes(outputs['instances'].pred_classes.tolist())
        #self.format_bbox(outputs['instances'].pred_boxes, outputs['instances'].pred_classes.tolist())
        self.format_masks(outputs['instances'].pred_masks.tolist(), outputs['instances'].pred_classes.tolist())

        # publish the message with the annotated image and the selected masks
        self.segmentation_pub.publish(self.msg)

        # publish the annotated image on a dedicated topic 
        annotated_img_msg = CompressedImage()
        # header and format
        annotated_img_msg.header.stamp = rospy.Time.now()
        annotated_img_msg.format = "jpeg"
        # actual image
        annotated_img_msg.data = np.array(cv2.imencode('.jpg', annotated_img_np)[1]).tostring()
        self.annotated_img_pub.publish(annotated_img_msg)

if __name__ == '__main__':
    # initialize and cleanup ROS node
    ic = ImageProcessing()
    rospy.init_node('image_processing', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS image processing module.")
    cv2.destroyAllWindows() # pretty sure this has become unuseful (but not harmful)
