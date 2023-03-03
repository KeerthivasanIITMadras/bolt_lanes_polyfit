#! /usr/bin/python

from polyfit.msg import abc_coeff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
import tf2_ros
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped
import math

import pandas as pd
import os
from typing import List

bridge = CvBridge()

rospy.init_node('DBSCAN')

pub_cluster = rospy.Publisher('/dbscan', Image, queue_size=1)

# tfBuffer = tf2_ros.Buffer()
# listener = tf2_ros.TransformListener(tfBuffer)

coeff = []



# x = 0
# y = 0
# z = 0
#
# prev_position = [0, 0, 0]
# prev_yaw = 0
#
#
# def position_callback(msg):
#    global x, y, z
#    x = msg.pose.position.x
#    y = msg.pose.position.y
#    z = msg.pose.position.z
#
#
# def find_position():
#    return [x, y, z]


# def get_transform():
#    global listener
#    global tfBuffer
#    transform = None
#    while not rospy.is_shutdown() and transform is None:
#        try:
#            transform = tfBuffer.lookup_transform(
#                "base_link", "map", rospy.Time())
#        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
#            rospy.logerr("Cant get transforms betweent the given frames")
#            continue
#        break
#    euler = euler_from_quaternion([transform.transform.rotation.x,
#                                   transform.transform.rotation.y,
#                                   transform.transform.rotation.z,
#                                   transform.transform.rotation.w])
#    yaw = euler[2]
#    return yaw


# def memory_filtering(xy: np.ndarray):
#    global prev_position
#    global prev_yaw
#    new_position = find_position()
#    current_yaw = get_transform()
#    change_yaw = current_yaw-prev_yaw
#    distance_x = abs(new_position[0]-prev_position[0])
#    distance_y = new_position[1]-prev_position[1]
#    new_xy = []
#    for i in xy:
#        if i[0] > distance_x:
#            new_xy.append([i[0]*math.cos(change_yaw)+i[1]*math.sin(change_yaw),
#                          i[1]*math.cos(change_yaw)-i[0]*math.sin(change_yaw)])
#    if len(new_xy) > 0:
#        new_xy = np.array(new_xy) - np.array([distance_x, distance_y])
#        return new_xy
#    return np.array([])


class Polynomial:

    def __init__(self):
        self.x_offset = -2.0#Both these shift the new created frame to base_footprint of the vehicle
        self.y_offset = 10.0#
        self.scale = 15
        self.pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)
        self.pub_mem = rospy.Publisher("/poly_viz_mem", Marker, queue_size=1)
        self.pub_poly = rospy.Publisher("/poly", abc_coeff, queue_size=10)
        # here the for previous part initially we will set it to some default polynomial with very low confidence
        self.prev_poly = np.empty((1, 3))
        self.prev_confidence = 0

    def poly_value(self, coeff, value):
        return coeff[0]*value*value+coeff[1]*value+coeff[2]

    def poly_viz(self):
        global coeff
        global memory_coeff
        line_strip = Marker()
        line_strip.header.frame_id = "base_link"
        line_strip.header.stamp = rospy.Time.now()  # 0 for add
        line_strip.pose.orientation.w = 1
        line_strip.type = 4  # 4 for line strip
        line_strip.scale.x = 0.1
        line_strip.color.b = 1.0
        pub = self.pub
        line_strip.color.a = 1.0
        range = 5
        resolution = 0.1
        i = 0
        for abc in coeff:
            line_strip.id = i
            line_strip.lifetime = rospy.Duration(0.2)
            x = 0
            while x <= range:
                point = Point()
                point.x = x
                point.y = self.poly_value(abc, x)
                point.z = 0
                line_strip.points.append(point)
                x = x+resolution
                i = i+1
            pub.publish(line_strip)
            line_strip.points.clear()
        if len(coeff) != 0:
            message = abc_coeff()
            coeff = np.array(coeff)
            message.a = coeff[:, 0].tolist()
            message.b = coeff[:, 1].tolist()
            message.c = coeff[:, 2].tolist()
            self.pub_poly.publish(message)
            coeff = coeff.tolist()

    def poly_find(self, xy: np.ndarray):
        # todo , make it just for one polynomial lane at time
        xy_g = self.img_to_world(xy)
        polynomial = list(np.polyfit(xy_g[:, 0], xy_g[:, 1], 2))
        return np.array(polynomial)
        # if self.color_coeff == 1:
        #    global coeff
        #    if polynomial[1] < 5 and polynomial[1] > -5:
        #        if polynomial[0] < 2 and polynomial[0] > -2:
        #            coeff.append(polynomial)
        # else:
        #    global memory_coeff
        #    if polynomial[1] < 5 and polynomial[1] > -5:
        #        if polynomial[0] < 2 and polynomial[0] > -2:
        #            memory_coeff.append(polynomial)

    def img_to_world(self, xy: np.ndarray):
        if len(xy) > 0:
            return xy/self.scale - np.array([self.x_offset, self.y_offset])
        return np.array([])

    def world_to_img(self, xy):
        if len(xy) > 0:
            return (xy+np.array([self.x_offset, self.y_offset]))*self.scale
        return np.array([])

    def r_square(self, poly: np.ndarray, cluster_pts: np.ndarray):
        sq_error = 0
        y_var = np.var(cluster_pts[:, 1])

        for x, y in cluster_pts:
            sq_error += (self.poly_value(x)-y)**2

        return (1 - (sq_error/y_var))

    def find_confidence(self, poly: np.ndarray, cluster_pts: np.ndarray, prev_lane: np.ndarray):
        # prev_lane is the lane poly coeff which is closest to the current lane
        if len(poly) != 3:
            raise ValueError("Polynomial has ", len(
                poly), " coefficients, expected 3")

        conf = self.r_square(poly, cluster_pts)

        # find the intercept difference
        # check which axis is the intercept of:
        diff = float(abs(prev_lane[2] - poly[2]))
        scale = max(0, 1-(diff/2.))
        conf = conf*scale
        return conf

    def prediction(self, prev_poly: np.ndarray, current_poly: np.ndarray, cluster_pts: np.ndarray):
        # Here i am writing it under the assumption that each object of polynomial corresponds to one lane
        current_confidence = self.find_confidence(
            current_poly, cluster_pts, prev_poly)
        predicted_poly = (current_confidence*current_poly +
                          self.prev_confidence*prev_poly)/(current_confidence+self.prev_confidence)
        self.prev_confidence = current_confidence*0.9
        self.prev_poly = current_poly
        return predicted_poly


def image_callback(msg):
    global coeff
    global pub_cluster
    polynomial_object = Polynomial(1)

    try:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    blank_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # blank_img_memory = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # if len(memory) > 0:
    #    memory = np.round(polynomial_object.world_to_img(
    #        memory_filtering(memory))).astype(int)
    #    db_memory = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(memory)
    #    core_samples_mask_memory = np.zeros_like(db_memory.labels_, dtype=bool)
    #    core_samples_mask_memory[db_memory.core_sample_indices_] = True
    #    lables_memory = db_memory.labels_
    #    unique_labels_memory = set(lables_memory)
#
    #    # for polynomials from memory
    #    for k, col in zip(unique_labels_memory, colors):
    #        if k == -1:
    #            col = (0, 0, 0)
    #        class_member_mask = (lables_memory == k)
    #        xy_core = memory[class_member_mask & core_samples_mask_memory]
    #        xy_non_core = memory[class_member_mask & ~core_samples_mask_memory]
    #        xy = np.concatenate([xy_core, xy_non_core])
    #        if len(xy) > 10:
    #            for i in xy:
    #                cv2.circle(blank_img_memory, tuple(
    #                    [int(i[1]), int(i[0])]), 0, col, -1)
    #            polynomial_object_mem.poly_find(xy)

    # np.findnonzero , np.find(img > 128)
    indexes_points = []
    for index, element in np.ndenumerate(img):
        if element > 128:
            indexes_points.append(tuple([index[0], index[1]]))

    indexes_points = np.array(indexes_points)
    # This is for the normal indices which we get at the correct time stamp
    db = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(indexes_points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    # what about no of lanes more than 4

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = (0, 0, 0)
        class_member_mask = (labels == k)
        xy_core = indexes_points[class_member_mask & core_samples_mask]
        xy_non_core = indexes_points[class_member_mask & ~core_samples_mask]
        xy = np.concatenate([xy_core, xy_non_core])
        if len(xy) > 10:
            for i in xy:
                cv2.circle(blank_img, tuple(
                    [int(i[1]), int(i[0])]), 0, col, -1)
            polynomial_object.poly_find(xy)
        xy_g = polynomial_object.img_to_world(xy)
        # if memory.size == 0:
        #     memory = xy_g
        # else:
        #     memory = np.concatenate((memory, xy_g))

    # prev_position = find_position()
    # prev_yaw = get_transform()
    pub_cluster.publish(bridge.cv2_to_imgmsg(blank_img, "passthrough"))
    # pub_cluster_memory.publish(
    #    bridge.cv2_to_imgmsg(blank_img_memory, "passthrough"))

    # combine the coefficients
    # new_coeff = []
    # for abc_mem in memory_coeff:
    #    c = 0
    #    for abc in coeff:
    #        if abs(abc_mem[0]-abc[0]) < 0.8 and abs(abc_mem[1]-abc[1]) < 0.8 and abs(abc_mem[2]-abc[2]) < 0.8:
    #            new_coeff.append(abc)
    #            c = c + 1
    #            break
    #    if c == 0:
    #        new_coeff.append(abc_mem)
    # coeff, new_coeff = new_coeff, coeff
    # if len(coeff) > 0:
    #    polynomial_object.poly_viz()
    # new_coeff = []
    coeff = []
    memory_coeff = []


def main():
    global tfBuffer
    global listener
    image_topic = "top_view"
    pose_topic = "/zed2i/zed_node/pose"

    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(pose_topic, PoseStamped, position_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
