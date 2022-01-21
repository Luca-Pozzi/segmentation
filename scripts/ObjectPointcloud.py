#!/home/luca/tiago_public_ws/src/segmentation/detectron_venv/bin/python3

# This script defines a subscriber node that reads the info published on 
# /output/segmentation and returns the pointcloud of every segmented object.

# Python libs
import numpy as np
import cv2
import sys

# ROS libs
import rospy
import message_filters

# ROS messages
from sensor_msgs.msg import PointCloud2, CompressedImage, CameraInfo
from sensor_msgs import point_cloud2
from segmentation.msg import SegmentationOutcome, ObjCloud
 
# my scripts
from vision_utils.decoder import decode_CompressedImage_depth, decode_CompressedImage_RGB
from img_segmentation import predictor_configuration, segment_image

# constant values and params
#SELECTED_CLASSES = [39, 40, 41, 60] # ['bottle', 'wine glass', 'cup', 'dining table']
VERBOSE = True
DISPLAY = True
CNN_MODEL_NAME = "mask_rcnn_R_50_FPN_3x"
SELECTED_CLASSES = [39]
SELECTED_CLASS_NAMES = ['bottle']

class ObjectCoordinates:
    def __init__(self):
        print('Initializing the CNN...', end = ' ')
        sys.stdout.flush()
        self.predictor_cfg, self.predictor = predictor_configuration(model_name = CNN_MODEL_NAME,
                                                                    score_threshold = rospy.get_param("/detection_score_threshold"))
        self.annotated_img = None
        print('Done.')

        print('Waiting for camera parameters...', end = ' ')
        sys.stdout.flush()
        # Get the parameters of the intrinsic matrix
        camera_info = rospy.wait_for_message('/camera_params', CameraInfo)
        self.fx = camera_info.K[0]
        self.fy = camera_info.K[4]
        self.cx = camera_info.K[2]
        self.cy = camera_info.K[5]
        self.S = camera_info.K[1]
        self.height = camera_info.height
        self.width = camera_info.width
        print('Done.')

        print('Define publishers and subscribers...', end = ' ')
        sys.stdout.flush()
        # initialize the publisher to broadcast the pointcloud, the geometrical features and the 
        # associated class 
        self.objcloud_pub = rospy.Publisher("/segmentation_result/cloud/semantic", 
                                            ObjCloud, 
                                            queue_size = 1
                                            )
        # initialize a subscriber that receives the images from the depth camera
        depth_sub = message_filters.Subscriber("/depth_image", CompressedImage)
        # initialize the subscriber node that receives the masks of the segmented objects
        rgb_sub = message_filters.Subscriber('/rgb_image', CompressedImage)
        # synchronize the subscribers and define a unique callback
        rgbd_sub = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub],
                                                                20, 0.02, allow_headerless = False
                                                                )
        rgbd_sub.registerCallback(self.mask_pointcloud)
        print('Done.\nThe node is operating.')

    def mask_pointcloud(self, rgb_image, depth_image):
        
        # Read the image from the CompressedImage messages
        rgb_img = decode_CompressedImage_RGB(rgb_image)
        depth_data, __ = decode_CompressedImage_depth(depth_image)
        
        # Segment the image
        self.annotated_img, outputs = segment_image(rgb_img, self.predictor_cfg, self.predictor)

        # Apply the masks to the pointcloud to get the pointclouds (and/or the features) of the interesting objects
        predicted_classes = outputs['instances'].pred_classes.tolist()
        predicted_masks = outputs['instances'].pred_masks.tolist()
        for label, mask in zip(predicted_classes, predicted_masks):
            if label in SELECTED_CLASSES:
                # Apply the mask to the pointcloud (where the mask is True, the data are hidden).
                mask = np.array(mask)
                masked_depth_data = np.ma.array(depth_data, mask = ~mask) 
                
                # Transform the points into 3D spatial coordinates.
                indexes = np.where(mask)
                
                uv = np.vstack((indexes[1], indexes[0])).T
                z = masked_depth_data.compressed()
                points = self.uvz_to_xyz(uv = uv,
                                         z = z,
                                         fx = self.fx, fy = self.fy, cx = self.cx, cy = self.cy
                                         )

                ''' does this really make sense?
                # Filter out the outliers
                neighbors = 100
                std_multiplier = 0.1 # damn low
                for point in points:
                    pass
                '''

                # Format the message to be published.
                
                objcloud = ObjCloud()
                objcloud.header.stamp = rospy.Time.now()
                objcloud.header.frame_id = depth_image.header.frame_id
                pcl = point_cloud2.create_cloud_xyz32(depth_image.header, 
                                                      np.reshape(points, 
                                                                 newshape = (-1, 3)
                                                                 )
                                                      )
                objcloud.cloud = pcl
                objcloud.obj_class = SELECTED_CLASS_NAMES[SELECTED_CLASSES.index(label)]
                self.objcloud_pub.publish(objcloud)

    @staticmethod
    def uvz_to_xyz(uv, z, fx, fy, cx, cy, S = 0):
        ##########
        # from: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
        # uv:   [list] pixel coordinates of the point in the image
        # z:    [float] depth value in the same measurement unit of the output
        # K:    [list[list]] intrinsic camera matrix
        ##########
        # nan is the value of keypoints that are not in the image: remove them from the vector
        # to send to matmul, store their index and re-add them to the output
        # TODO: make this part a little bit more good-looking (or maybe move outside the func)
        original_length = len(uv)
        where_are_nans = np.array([np.isnan(np.sum(element)) for element in uv], dtype = bool)
        uv = uv[~where_are_nans]
        z = z[~where_are_nans]

        # format the arrays 
        z = z[:, np.newaxis]                                        # (N, 1)
        ones = np.ones((z.shape))                                   # (N, 1)
        #print(uv.shape, z.shape, ones.shape)
        uv = np.hstack((uv, ones, np.reciprocal(z)))                # (N, 4)
        # attach a dummy dimension so that matmul sees it as a stack of (4, 1) vectors
        uv = np.expand_dims(uv, axis = 2)                           # (N, 4, 1)
        
        # invert the intrinsic matrix
        #fx, S, cx, __, fy, cy = *K[0], *K[1]
        K_inv = [   [1/fx, -S/(fx*fy), (S*cy-cx*fy)/(fx*fy), 0],
                    [0, 1/fy, -cy/fy, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]
        # compute the spacial 3D coordinates for the points
        xyz = z[:, np.newaxis] * np.matmul(K_inv, uv)   # (N, 4, 1)
        xyz = xyz[:, :3].reshape(-1, 3)                # (N, 3)
        
        # re-add missing keypoint in their original position
        # TODO: make this part a little bit more good-looking (or maybe move outside the func)
        xyz_complete = np.empty((original_length, 3))
        xyz_complete[~where_are_nans, :] = xyz
        xyz_complete[where_are_nans] = np.array([np.nan, np.nan, np.nan])
        return xyz_complete

if __name__ == '__main__':
    rospy.init_node('coordinates_finder', anonymous = True)
    finder = ObjectCoordinates()
    try:
        while not rospy.is_shutdown():
            if finder.annotated_img is not None and DISPLAY:
                cv2.imshow("Annotated image", finder.annotated_img)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Shutting down the coordinate finder module.")

    cv2.destroyAllWindows()