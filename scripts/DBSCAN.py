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


# Instantiate CvBridge
bridge = CvBridge()


pub_cluster = rospy.Publisher('/dbscan', Image, queue_size=1)


coeff = []

memory = np.array([])


x = 0
y = 0
z = 0

prev_position = [0, 0, 0]


def position_callback(msg):
    global x, y, z
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z


def find_position():
    return [x, y, z]


def memory_filtering(xy):
    global prev_position
    new_position = find_position()
    distance_x = abs(new_position[0]-prev_position[0])
    distance_y = new_position[1]-prev_position[1]
    new_xy = []
    for i in xy:
        if i[0] > distance_x and i[1]-distance_y < 4 and i[1]-distance_y > -4:
            new_xy.append(i)
    if len(new_xy) > 0:
        new_xy = np.array(new_xy) - np.array([distance_x, distance_y])
        return new_xy
    return np.array([])


class Polynomial:

    def __init__(self):
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15
        self.pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)
        self.pub_poly = rospy.Publisher("/poly", abc_coeff, queue_size=10)

    def poly_value(self, coeff, value):
        return coeff[0]*value*value+coeff[1]*value+coeff[2]

    def poly_viz(self, coeff):
        line_strip = Marker()
        line_strip.header.frame_id = "base_link"
        line_strip.header.stamp = rospy.Time.now()  # 0 for add
        line_strip.pose.orientation.w = 1
        line_strip.type = 4  # 4 for line strip
        line_strip.scale.x = 0.1
        line_strip.color.b = 1.0
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
            self.pub.publish(line_strip)
            line_strip.points.clear()
        if len(coeff) != 0:
            message = abc_coeff()
            coeff = np.array(coeff)
            message.a = coeff[:, 0].tolist()
            message.b = coeff[:, 1].tolist()
            message.c = coeff[:, 2].tolist()
            self.pub_poly.publish(message)
            coeff = coeff.tolist()

    def poly_find(self, xy):
        global coeff
        xy_g = self.img_to_world(xy)
        polynomial = np.polyfit(xy_g[:, 0], xy_g[:, 1], 2)
        if polynomial[1] < 5 and polynomial[1] > -5:
            if polynomial[0] < 2 and polynomial[0] > -2:
                coeff.append(polynomial)

    def img_to_world(self, xy):
        if len(xy) > 0:
            return xy/self.scale - np.array([self.x_offset, self.y_offset])
        return np.array([])

    def world_to_img(self, xy):
        if len(xy) > 0:
            return (xy+np.array([self.x_offset, self.y_offset]))*self.scale
        return np.array([])


def image_callback(msg):
    global coeff
    global pub_cluster
    global memory
    global prev_position
    global pub_poly

    polynomial_object = Polynomial()

    try:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    blank_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    if len(memory) > 0:
        memory = polynomial_object.world_to_img(memory_filtering(memory))

    indexes_points = []
    for index, element in np.ndenumerate(img):
        if element > 128:
            indexes_points.append(tuple([index[0], index[1]]))

    if len(indexes_points) == 0:
        memory = []
        return

    # indices to delete which fall within a specific radius

    indices_to_delete = []
    radius = 4
    k = []
    if len(memory) > 0:
        for i, coord1 in enumerate(indexes_points):
            for coord2 in memory:
                dist = np.sqrt((coord1[0]-coord2[0]) **
                               2 + (coord1[1]-coord2[1])**2)
                k.append(dist)
                if dist < radius:
                    indices_to_delete.append(coord1)
                    break
        indices_to_delete = np.array(indices_to_delete)

        for i in memory:
            result = ~np.isin(i, indices_to_delete)
            if result.all():
                indexes_points.append(tuple([int(i[0]), int(i[1])]))

    indexes_points = np.array(indexes_points)

    memory = np.array([])

    db = DBSCAN(eps=6, min_samples=25, algorithm='auto').fit(indexes_points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    # what about no of lanes more than 4
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = (0, 0, 0)
        class_member_mask = (labels == k)
        xy_core = indexes_points[class_member_mask & core_samples_mask]
        xy_non_core = indexes_points[class_member_mask & ~core_samples_mask]
        xy = np.concatenate([xy_core, xy_non_core])
        if len(xy) > 75:
            for i in xy:
                cv2.circle(blank_img, tuple(
                    [int(i[1]), int(i[0])]), 0, col, -1)
            polynomial_object.poly_find(xy)

        xy_g = polynomial_object.img_to_world(xy)
        if memory.size == 0:
            memory = xy_g
        else:
            memory = np.concatenate((memory, xy_g))
    prev_position = find_position()

    pub_cluster.publish(bridge.cv2_to_imgmsg(blank_img, "passthrough"))
    polynomial_object.poly_viz(coeff)
    coeff = []


def main():
    global coeff
    image_topic = "top_view"
    pose_topic = "/zed2i/zed_node/pose"

    rospy.init_node('DBSCAN')
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(pose_topic, PoseStamped, position_callback)

    rospy.spin()


if __name__ == '__main__':
    main()
