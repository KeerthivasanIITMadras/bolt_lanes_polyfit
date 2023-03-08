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

bridge = CvBridge()

rospy.init_node('DBSCAN')

# pub_cluster = rospy.Publisher('/dbscan', Image, queue_size=1)


class Polynomial:
    # polyfind, img to world and world to img hv been moved to class Lanes
    def __init__(self, abc):
        # Both these shift the new created frame to base_footprint of the vehicle
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15
        # here the for previous part initially we will set it to some default polynomial with very low confidence
        self.prev_poly = abc
        self.prev_confidence = 0  # todo give some low confidence
        self.centroid = None
        self.curr_poly = None
        self.points:np.ndarray = np.array([])

    def poly_value(self, value):
        # Here prev_poly corresponds to the estimated polynomial , since we are changing it in the prediction
        return self.prev_poly[0]*value*value + self.prev_poly[1]*value + self.prev_poly[2] 

    def img_to_world(self, xy: np.ndarray):
        if len(xy) > 0:
            return xy/self.scale - np.array([self.x_offset, self.y_offset])
        return np.array([])

    def poly_find(self):
        # todo , make it just for one polynomial lane at time
        xy_g = self.img_to_world(self.points)
        polynomial = list(np.polyfit(xy_g[:, 0], xy_g[:, 1], 2))
        self.curr_poly = np.array(polynomial)

    def r_square(self, poly: np.ndarray, cluster_pts: np.ndarray):
        sq_error = 0
        y_var = np.var(cluster_pts[:, 1])

        for x, y in cluster_pts:
            sq_error += (poly(x)-y)**2

        return (1 - (sq_error/y_var))

    def find_confidence(self, poly: np.ndarray):
        # prev_lane is the lane poly coeff which is closest to the current lane
        conf = self.r_square(poly, self.points)

        # find the intercept difference
        # check which axis is the intercept of:
        diff = float(abs(self.prev_poly[2] - poly[2]))
        scale = max(0, 1-(diff/2.))
        conf = conf*scale
        return conf

    def prediction(self):
        # Here i am writing it under the assumption that each object of polynomial corresponds to one lane
        current_confidence = self.find_confidence(self.curr_poly)
        predicted_poly = (current_confidence*self.curr_poly +
                          self.prev_confidence*self.prev_poly)/(current_confidence+self.prev_confidence)
        actual_confidence = self.find_confidence(predicted_poly)
        self.prev_confidence = actual_confidence*0.9
        self.prev_poly = predicted_poly
        return predicted_poly


class Lanes:
    def __init__(self) -> None:
        self.left_lane = Polynomial([0, 0, -1.5])
        self.mid_lane = Polynomial([0, 0, 1.5])
        self.right_lane = Polynomial([0, 0, 4.5])
        self.dummy_lane = Polynomial([0, 0, 0])
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15

        self.pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)
        self.pub_poly = rospy.Publisher("/poly", abc_coeff, queue_size=10)

    def img_to_world(self, xy: np.ndarray):
        if len(xy) > 0:
            return xy/self.scale - np.array([self.x_offset, self.y_offset])
        return np.array([])

    def world_to_img(self, xy):
        if len(xy) > 0:
            return (xy+np.array([self.x_offset, self.y_offset]))*self.scale
        return np.array([])

    def find_lanes(self, db, X: np.ndarray):
        '''
            combine clusters which are close to each other
            compares with prev lanes
        '''
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # All core points are true, else false
        core_samples_mask[db.core_sample_indices_] = True
        unique_labels = set(db.labels_)  # gets all the labels
        labels = db.labels_
        # label -1 which is for noisy points
        unique_labels.remove(-1)
        new_labels = {}
        # centroids = np.zeros((len(unique_labels), 2))
        centroids = {}  # dict of centroids of all lanes
        # find the centroids
        for k in unique_labels:
            if k == -1:
                continue
            class_member_mask = (labels == k)  # if label is k
            # if label is k and it is a core point
            xy_core = X[class_member_mask & core_samples_mask]
            # if label is k and it is not a core point
            xy_non_core = X[class_member_mask & ~core_samples_mask]
            # all points of label k
            xy = np.concatenate([xy_core, xy_non_core])
            # print(len(xy))
            new_labels[k] = xy
            centroids[k] = [0, 0]
            centroids[k][0] = sum(xy[:, 0])/xy.shape[0]
            centroids[k][1] = sum(xy[:, 1])/xy.shape[0]

        # combines the centroids close by
        unique_labels = list(unique_labels)
        for i in range(len(unique_labels)):
            if unique_labels[i] not in centroids.keys():
                continue
            for j in range(i+1, len(unique_labels)):
                if unique_labels[j] not in centroids.keys():
                    continue
                distance = np.sqrt(
                    (centroids[unique_labels[i]][0]-centroids[unique_labels[j]][0])**2 + (centroids[unique_labels[j]][1]-centroids[unique_labels[j]][1])**2)
                if distance < 1:
                    new_labels[i] = np.concatenate(
                        [new_labels[i], new_labels[j]])
                    # update centroids -> i
                    # no. of points in the ith cluster
                    n1 = len(unique_labels[i])
                    # no. of points in the jth cluster
                    n2 = len(unique_labels[j])
                    centroids[unique_labels[i]][0] = (
                        centroids[unique_labels[i]][0]*n1 + centroids[unique_labels[j]][0]*n2) / (n1+n2)
                    centroids[unique_labels[i]][1] = (
                        centroids[unique_labels[i]][1]*n1 + centroids[unique_labels[j]][1]*n2) / (n1+n2)

                    # delete centroids -> j
                    del centroids[unique_labels[j]]
                    del new_labels[unique_labels[j]]

        lane_coeff = [self.left_lane.prev_poly[2],
                      self.mid_lane.prev_poly[2], self.right_lane.prev_poly[2]]
        # final_3_lanes is a dictionary that will store clusters in each of the lanes
        self.left_lane.points = []
        self.mid_lane.points = []
        self.right_lane.points = []

        for i in new_labels.keys:
            # find the lane polynomial
            self.dummy_lane.points = new_labels[i]
            self.dummy_lane.poly_find()

            # d1, d2, d3 is distance of each clusters polynomial with the left mid and right lane
            d1 = abs(self.dummy_lane.curr_poly[2] - lane_coeff[0])
            d2 = abs(self.dummy_lane.curr_poly[2] - lane_coeff[1])
            d3 = abs(self.dummy_lane.curr_poly[2] - lane_coeff[2])

            # checking which lane is nearest to the given cluster polynomial and discarding clusters
            # with distance greater than 1.5 metres with all 3 clusters
            if(min(d1, d2, d3) == d1 and d1 < 1.5):
                # appending the clusters to final_3_lanes
                self.left_lane.points.concatenate(new_labels[i])
            elif(min(d1, d2, d3) == d2 and d2 < 1.5):
                self.mid_lane.points.concatenate(new_labels[i])
            elif(min(d1, d2, d3) == d3 and d3 < 1.5):
                self.right_lane.points.concatenate(new_labels[i])

    def poly_viz(self):
        coeff = [self.left_lane.curr_poly,
                 self.mid_lane.curr_poly, self.right_lane.curr_poly]
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
                point.y = abc[0]*x**2+abc[1]*x+abc[2]
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
    # still need to confirm this part


lanes = Lanes()


def image_callback(msg):
    global coeff
    global lanes

    try:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    indexes_points = []  # all the lane points
    for index, element in np.ndenumerate(img):
        if element > 128:
            # check if it is white
            # adds image pixel coordinates to the list
            indexes_points.append(tuple([index[0], index[1]]))

    indexes_points = np.array(indexes_points)
    # This is for the normal indices which we get at the correct time stamp
    db = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(
        indexes_points)  # clusters lane points
    lanes.find_lanes(db, indexes_points)

    # find the current poly coeff here
    lanes.left_lane.poly_find()
    lanes.mid_lane.poly_find()
    lanes.right_lane.poly_find()

    abc_left = lanes.left_lane.prediction()
    abc_mid = lanes.mid_lane.prediction()
    abc_right = lanes.right_lane.prediction()

    lanes.poly_viz()

    # pulish this
    lane_coeff = [list(abc_left), list(abc_mid), list(abc_right)]


def main():
    image_topic = "top_view"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
