#!/usr/bin/env python
# match the pointcloud sent through the filtering nodelet with their label

# ROS libs
import rospy
import message_filters # synchronize topics
# ROS messages
from sensor_msgs.msg import PointCloud2
from segmentation.msg import ObjCloud, ClassStamped	# custom messages, see the msg folder for further
                                        			# info about fields and data types

class SemanticCloud:
	def __init__(self):
		# subscribe to the topic where the PCL nodelets publishes the filtered pointclouds
		self.pointcloud_subscriber = message_filters.Subscriber("/object_outlier_removal/output", PointCloud2)
		# subscribe to a topic that allow to match the pointcloud timestamp with the class of the object that it represents
		self.class_info = message_filters.Subscriber("/segmentation_result/class", ClassStamped)
		# exploit message_filter to synchronize the two subscriber based on the headers in their messages
		self.pointcloud_with_class = message_filters.TimeSynchronizer([self.pointcloud_subscriber, self.class_info], 10)
		self.pointcloud_with_class.registerCallback(self.match_cloud_and_label) # define a common callback
		print('Subscribed to /object_outlier_removal/output and /segmentation_result/coordinates.' + 
			'Messages from the subscribed topic are going to be synchronized based on their timestamps.')
		# initialize the publisher to broadcast the pointcloud, the geometrical features and the associated class 
		self.objcloud_publisher = rospy.Publisher("/segmentation_result/cloud/semantic", ObjCloud, queue_size = 1)

	def match_cloud_and_label(self, cloud, pred_class):
		##########
		# Callback of pointcloud_subscriber and class_info (receiving synchronized messages from the two subscribers).
		##########
		# initialize the message with the information about the segmented pointcloud
		objcloud = ObjCloud()
		# set the header of the message
		try:
			assert cloud.header.stamp == pred_class.header.stamp
		except AssertionError:
			print("The timestamps of the class message and of the cloud are not the same:\nCloud:\n" + 
				str(cloud.header.stamp) + "Class:\n" + str(pred_class.header.stamp) + '\nHave you messed up with ObjectPointcloud.py or processing.py?')
		objcloud.header = cloud.header
		# save the info about the predicted class of object
		objcloud.obj_class = pred_class.obj_class
		# add the pointcloud to the message (currently unused)
		objcloud.cloud = cloud
		# publish the message
		self.objcloud_publisher.publish(objcloud)

if __name__ == "__main__":
	rospy.init_node("pointcloud_labeler", anonymous = True)
	geom = SemanticCloud()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print('Shutting down the pointcloud_labeler module.')
