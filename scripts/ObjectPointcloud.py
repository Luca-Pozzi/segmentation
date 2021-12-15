#!/home/luca/tiago_public_ws/src/segmentation/detectron_venv/bin/python3

# This script defines a subscriber node that reads the info published on 
# /output/segmentation and returns the pointcloud of every segmented object.

# Python libs
import numpy as np
import cv2
# ROS libs
import rospy
import message_filters

# ROS messages
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs import point_cloud2
from segmentation.msg import SegmentationOutcome, ClassStamped  # custom messages, see the msg folder for further
                                                                # info about fields and data types
# my scripts
from img_segmentation import predictor_configuration, segment_image

# constant values and params
#SELECTED_CLASSES = [39, 40, 41, 60] # ['bottle', 'wine glass', 'cup', 'dining table']
VERBOSE = True
CNN_MODEL_NAME = "mask_rcnn_R_50_FPN_3x"
SELECTED_CLASSES = [39]
SELECTED_CLASS_NAMES = ['bottle']

class ObjectCoordinates:
    def __init__(self):
        if VERBOSE:
            print('Starting the nodes.')
        self.predictor_cfg, self.predictor = predictor_configuration(model_name = CNN_MODEL_NAME,
                                                                    score_threshold = rospy.get_param("/detection_score_threshold"))
        # initialize the publishers used to broadcast the results
        self.class_publisher = rospy.Publisher('/segmentation_result/class', ClassStamped, queue_size = 5)
        self.cloud_publisher = rospy.Publisher('/segmentation_result/cloud', PointCloud2, queue_size = 5)
        self.annotated_img_pub = rospy.Publisher('/segmentation_result/annotated_img/compressed', CompressedImage, queue_size = 1)
        # initialize a subscriber that receives the images from the depth camera
        self.depth_subscriber = message_filters.Subscriber("/pointcloud", PointCloud2)
        # initialize the subscriber node that receives the masks of the segmented objects
        self.rgb_subscriber = message_filters.Subscriber('/rgb_image', CompressedImage)
        # synchronize the subscribers and define a unique callback
        self.filter_pointcloud = message_filters.ApproximateTimeSynchronizer([self.depth_subscriber, self.rgb_subscriber], 10, slop = 0.1)
        self.filter_pointcloud.registerCallback(self.mask_pointcloud)
        print('Done')

    def mask_pointcloud(self, pointcloud2, rgb_image):
        # Conversion to cv2 image
        np_arr = np.fromstring(rgb_image.data, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # segment the image
        annotated_img_np, outputs = segment_image(img_np, self.predictor_cfg, self.predictor)
        #t = time.time()
        # publish the annotated image
        annotated_img_msg = CompressedImage() 
        # header and format
        annotated_img_msg.header.stamp = rospy.Time.now()
        annotated_img_msg.format = "jpeg"
        # actual image
        annotated_img_msg.data = np.array(cv2.imencode('.jpg', annotated_img_np)[1]).tostring()
        self.annotated_img_pub.publish(annotated_img_msg)
        #print("Publishing the annotated image took " + str(time.time() - t))   # it takes up to 0.01s
                                                                                # think about making this part optional, even at runtime through a param
        # apply the masks to the pointcloud to get the pointclouds (and/or the features) of the interesting objects
        predicted_classes = outputs['instances'].pred_classes.tolist()
        predicted_masks = outputs['instances'].pred_masks.tolist()
        for label, mask in zip(predicted_classes, predicted_masks):
            if label in SELECTED_CLASSES:
                bool_mask = np.stack([np.array(mask)] * 3, axis = -1)
                
                '''
                TODO:   move the following two lines OUT of the for loop. This won't probably impact on the tiago_pouring behavior as in that application
                        there is tipically a single object belonging to the selected class
                '''
                # get an iterable object from the pointcloud and use it to fill a numpy array of points (faster than point_cloud2.read_points_list)
                depth_points_iterable = point_cloud2.read_points(pointcloud2, skip_nans = False, field_names = ("x", "y", "z"))
                depth_points_array = np.reshape(np.array([point for point in depth_points_iterable]), newshape = (pointcloud2.height, pointcloud2.width, 3))
                
                # apply the mask to the pointcloud
                masked_depth_points = np.ma.array(depth_points_array, mask = ~bool_mask) # were the mask is 1 the data are hidden
                ### publish PointCloud2 message
                time_stamp = rospy.Time.now()   # the same timestamp is assigned to both the cloud and the mask messages to allow synchronizing them later
                pcl = point_cloud2.create_cloud_xyz32(pointcloud2.header, np.reshape(masked_depth_points.compressed(), newshape = (-1, 3)))
                pcl.header.stamp = time_stamp       # update the time_stamp
                self.cloud_publisher.publish(pcl)   # publish the message
                ### publish ClassStamped message
                class_msg = ClassStamped()
                class_msg.header.stamp = time_stamp
                class_msg.obj_class = SELECTED_CLASS_NAMES[SELECTED_CLASSES.index(label)] 
                self.class_publisher.publish(class_msg)

if __name__ == '__main__':
    rospy.init_node('coordinates_finder', anonymous = True)
    finder = ObjectCoordinates()
    try:
        rospy.spin()    
    except KeyboardInterrupt:
        print("Shutting down the coordinate finder module.")