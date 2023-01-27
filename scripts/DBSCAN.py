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
import tf2_ros


# Instantiate CvBridge
bridge = CvBridge()


pub_cluster = rospy.Publisher('/dbscan', Image, queue_size=1)

coeff = []

memory = np.array([])

tf_buffer = 0
tf_listener = 0

time_prev = rospy.Time(0)

transform_stamped_1 = 0

frame_id_1 = "odom"
frame_id_2 = "map"

g =Odometry()
def velocity_callback(msg):
    global vx, vy, wz
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    wz = msg.twist.twist.angular.z
    #print(f"{vx}  {vy}  {wz}")
    print(msg.twist.twist.linear.x)


def memory_filtering(xy):
    global time_prev
    global transform_stamped_1
    global frame_id_1
    global frame_id_2
    #global tf_buffer
    #global tf_listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    '''delta_t = time_now - time_prev
    distance_x = vx*delta_t
    distance_y = vy*delta_t
    theta = wz*delta_t
    rotation_matrix = [[math.cos(theta), math.sin(theta)], [
        math.sin(-theta), math.cos(theta)]]
    new_xy = []
    for i in xy:
        if i[0] > distance_x and i[1] > distance_y:
            i = i*rotation_matrix
            new_xy.append(i)
    if len(new_xy):
        new_xy = np.array(new_xy) - np.array([distance_x, distance_y])
        return new_xy
    return np.array([])'''

    #time_stamp_1 = rospy.Time(time_prev)
    time_stamp_2 = rospy.Time(0)

    try:
        transform_stamped_2 = tf_buffer.lookup_transform(
            frame_id_1, frame_id_1, time_stamp_2)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("Error trying to get transform between time stamps")

    if transform_stamped_1 != 0:
        trans = transform_stamped_1.transform.translation
        rot = transform_stamped_1.transform.rotation
        # print(trans)
    trans2 = transform_stamped_2.transform.translation
    rot2 = transform_stamped_2.transform.rotation

    # print(rot2)
    return np.array(())


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
        #print(type((xy+np.array([self.x_offset, self.y_offset]))*self.scale))
        if len(xy) > 0:
            return (xy+np.array([self.x_offset, self.y_offset]))*self.scale
        return np.array([])


def image_callback(msg):
    global coeff
    global pub_cluster
    global time_prev
    global memory
    global frame_id_1
    global frame_id_2
    global transform_stamped_1

    polynomial_object = Polynomial()
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

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
    for i in memory:
        indexes_points.append(tuple(i))
    indexes_points = list(set(tuple(indexes_points)))
    indexes_points = np.array(indexes_points)
    if indexes_points.size == 0:
        return

    memory = np.array([])

    db = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(indexes_points)
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
    time_prev = rospy.Time(0)

    try:
        transform_stamped_1 = tf_buffer.lookup_transform(
            frame_id_2, frame_id_1, time_prev)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("Error trying to get transform between time stamps")
        print("Is this working")

    # print(time_prev)
    pub_cluster.publish(bridge.cv2_to_imgmsg(blank_img, "passthrough"))
    polynomial_object.poly_viz(coeff)
    coeff = []


def main():
    image_topic = "top_view"
    velocity_topic = "/zed2i/zed_node/odom"

    rospy.init_node('DBSCAN')
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(velocity_topic, Odometry, velocity_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
