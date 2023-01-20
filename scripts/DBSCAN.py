#! /usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry


# Instantiate CvBridge
bridge = CvBridge()


pub_cluster = rospy.Publisher('/dbscan', Image, queue_size=1)

coeff = []

memory_coordinates = []

time1 = 0


class Polynomial:

    def __init__(self):
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15
        self.pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)

    def poly_value(self, coeff, value):
        return coeff[0]*value*value+coeff[1]*value+coeff[2]

    def poly_viz(self, coeff):
        line_strip = Marker()
        line_strip.header.frame_id = "map"
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

    def poly_find(self, xy, flag):
        global coeff
        x_g = []
        y_g = []
        if not flag:
            x_g = xy[:, 0]/self.scale - self.x_offset
            y_g = xy[:, 1]/self.scale - self.y_offset
        else:
            x_g = xy[:, 0]
            y_g = xy[:, 1]
        polynomial = np.polyfit(x_g, y_g, 2)
        if polynomial[1] < 5 and polynomial[1] > -5:
            if polynomial[0] < 2 and polynomial[0] > -2:
                coeff.append(polynomial)


class Memory:
    def __init__(self):
        velocity_topic = "/zed2i/zed_node/odom"
        rospy.Subscriber(velocity_topic, Odometry, self.velocity_callback)
        self.odom = Odometry()
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15

    def velocity_callback(self, odom):
        self.odom.twist.twist.linear.x = odom.twist.twist.linear.x
        self.odom.twist.twist.linear.y = odom.twist.twist.linear.y

    def update(self, memory, time_prev):
        time_now = rospy.get_rostime().secs
        delta_time = time_now - time_prev
        distance_moved_x = self.odom.twist.twist.linear.x*delta_time
        distance_moved_y = self.odom.twist.twist.linear.y*delta_time
        self.memory = memory/self.scale - \
            np.array([self.x_offset, self.y_offset])
        self.memory = self.memory[(self.memory[:, 0] >=
                                  distance_moved_x) and (self.memory[:, 1] >= distance_moved_y)]
        self.memory = self.memory - \
            np.array([distance_moved_x, distance_moved_y])

    def points_taken(self, xy):
        points = np.array([])
        for i in self.memory:
            for j in xy:
                if abs(i[0, 0]-j[0, 0]) < 0.1 and abs(i[0, 1]-j[0, 1]):
                    points = np.concatenate(points, i)
        return points, True


def image_callback(msg):
    global coeff
    global pub_cluster
    global memory_coordinates
    global time1

    try:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    blank_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    indexes_points = []
    for index, element in np.ndenumerate(img):
        if element != 0:
            indexes_points.append([index[0], index[1]])
    indexes_points = np.array(indexes_points)
    if indexes_points.size == 0:
        return

    db = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(indexes_points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)

    polynomial_object = Polynomial()
    memory_object = Memory()

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
                cv2.circle(blank_img, tuple([i[1], i[0]]), 0, col, -1)
            if memory_coordinates == []:
                memory_coordinates = xy
                polynomial_object.poly_find(xy, False)
                time1 = rospy.get_rostime().secs
            else:
                memory_object.update(memory_coordinates, time1)
                memory_coordinates = xy
                xy, flag = memory_object.points_taken(xy)
                time1 = rospy.get_rostime().secs
            polynomial_object.poly_find(xy, flag)

    pub_cluster.publish(bridge.cv2_to_imgmsg(blank_img, "passthrough"))
    polynomial_object.poly_viz(coeff)
    coeff = []


def main():
    image_topic = "top_view"

    rospy.init_node('DBSCAN')
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
