#! /usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# Instantiate CvBridge
bridge = CvBridge()

pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)


x_offset = -2.0
y_offset = 10.0
scale = 15

coeff = []


def poly_value(coeff, value):
    return coeff[0]*value*value+coeff[1]*value+coeff[2]


def poly_viz(coeff):
    global pub
    line_strip = Marker()
    line_strip.header.frame_id = "odom"
    line_strip.header.stamp = rospy.Time.now()  # 0 for add
    line_strip.pose.orientation.w = 1
    line_strip.type = 4  # 4 for line strip
    line_strip.scale.x = 0.1
    line_strip.color.b = 1.0
    line_strip.color.a = 1.0
    range = 4
    resolution = 0.1
    i = 0
    print(len(coeff))
    for abc in coeff:
        line_strip.id = i
        x = -range
        while x <= range:
            point = Point()
            point.x = x
            point.y = poly_value(abc, x)
            point.z = 0
            line_strip.points.append(point)
            x = x+resolution
        i = i+1
        pub.publish(line_strip)
        line_strip.points.clear()
    
    


def poly_find(xy):
    global coeff
    x_g = xy[:, 0]/scale - x_offset
    y_g = xy[:, 1]/scale - y_offset
    polynomial = np.polyfit(x_g, y_g, 2)

    # Here the abc parameters are in world coordinates
    coeff.append(polynomial)  # polynomial is a,b,c


def image_callback(msg):
    global pub
    global coeff
    try:
        # Convert your ROS Image message to OpenCV2
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    indexes_points = []
    for index, element in np.ndenumerate(img):
        if element != 0:
            indexes_points.append([index[0], index[1]])

    indexes_points = np.array(indexes_points)

    X = indexes_points
    if X.size == 0:
        return
    db = DBSCAN(eps=15, min_samples=30, algorithm='auto').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    unique_labels = set(labels)

    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = (0, 0, 0)
        class_member_mask = (labels == k)
        xy_core = X[class_member_mask & core_samples_mask]
        xy_non_core = X[class_member_mask & ~core_samples_mask]
        xy = np.concatenate([xy_core, xy_non_core])
        poly_find(xy)

    poly_viz(coeff)
    coeff = []


def main():
    rospy.init_node('DBSCAN')
    # Define your image topic
    image_topic = "top_view"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()


if __name__ == '__main__':
    main()
