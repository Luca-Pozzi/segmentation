#!/usr/bin/env python
# This nodes displays a window showing the (almost) real-time annotated image coming from the
# processing.py node (or any other node publishing to /segmentation_result)

# Python libs
import numpy as np
import cv2
# ROS libs
import rospy
# ROS messages
from segmentation.msg import SegmentationOutcome    # custom message

class Visualizer:
    def __init__(self):
        # subscribe to the topic where the outcome of the segmentation of the result is published
        rospy.Subscriber("/segmentation_result", SegmentationOutcome, self.visualize)

    def visualize(self, segmentation_output):
        # conversion to cv2 image
        np_arr = np.fromstring(segmentation_output.annotated_img.data, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # show the cv2 image and place the window in the specified position 
        cv2.imshow("That's what TIAGo sees", img_np)
        cv2.waitKey(1) #[ms]

if __name__ == '__main__':
    # initialize a ROS node
    rospy.init_node('visualizer', anonymous = True)
    visual = Visualizer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS image processing module.")
    cv2.destroyAllWindows()