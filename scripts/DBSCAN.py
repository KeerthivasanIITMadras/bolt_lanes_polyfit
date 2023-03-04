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


coeff = []

class Polynomial:
    # polyfind, img to world and world to img hv been moved to class Lanes
    def __init__(self, abc):
        # Both these shift the new created frame to base_footprint of the vehicle
        self.x_offset = -2.0
        self.y_offset = 10.0
        self.scale = 15
        self.pub = rospy.Publisher("/poly_viz", Marker, queue_size=100)
        self.pub_poly = rospy.Publisher("/poly", abc_coeff, queue_size=10)
        # here the for previous part initially we will set it to some default polynomial with very low confidence
        self.prev_poly = abc
        self.prev_confidence = 0  # todo give some low confidence
        self.centroid = None
        self.coeff = None

    def poly_value(self, value):
        return self.coeff[0]*value*value + self.coeff[1]*value + self.coeff[2]

    # def poly_viz(self):
    #     global coeff
    #     line_strip = Marker()
    #     line_strip.header.frame_id = "base_link"
    #     line_strip.header.stamp = rospy.Time.now()  # 0 for add
    #     line_strip.pose.orientation.w = 1
    #     line_strip.type = 4  # 4 for line strip
    #     line_strip.scale.x = 0.1
    #     line_strip.color.b = 1.0
    #     pub = self.pub
    #     line_strip.color.a = 1.0
    #     range = 5
    #     resolution = 0.1
    #     i = 0
    #     for abc in coeff:
    #         line_strip.id = i
    #         line_strip.lifetime = rospy.Duration(0.2)
    #         x = 0
    #         while x <= range:
    #             point = Point()
    #             point.x = x
    #             point.y = self.poly_value(abc, x)
    #             point.z = 0
    #             line_strip.points.append(point)
    #             x = x+resolution
    #             i = i+1
    #         pub.publish(line_strip)
    #         line_strip.points.clear()
    #     if len(coeff) != 0:
    #         message = abc_coeff()
    #         coeff = np.array(coeff)
    #         message.a = coeff[:, 0].tolist()
    #         message.b = coeff[:, 1].tolist()
    #         message.c = coeff[:, 2].tolist()
    #         self.pub_poly.publish(message)
    #         coeff = coeff.tolist()
    # # still need to confirm this part

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

    def prediction(self, current_poly: np.ndarray, cluster_pts: np.ndarray):
        # Here i am writing it under the assumption that each object of polynomial corresponds to one lane
        current_confidence = self.find_confidence(
            current_poly, cluster_pts, self.prev_poly)
        predicted_poly = (current_confidence*current_poly +
                          self.prev_confidence*self.prev_poly)/(current_confidence+self.prev_confidence)
        self.prev_confidence = current_confidence*0.9
        self.prev_poly = current_poly
        return predicted_poly

class Lanes:
    def __init__(self) -> None:
        self.left_lane = Polynomial('''give the initial a,b,c here''')
        self.mid_lane = Polynomial('''give the initial a,b,c here''')
        self.right_lane = Polynomial('''give the initial a,b,c here''')
        self.dummy_lane = Polynomial([0, 0, 0])
        
    def poly_find(self, xy: np.ndarray):
        # todo , make it just for one polynomial lane at time
        xy_g = self.img_to_world(xy)
        polynomial = list(np.polyfit(xy_g[:, 0], xy_g[:, 1], 2))
        return np.array(polynomial)

    def img_to_world(self, xy: np.ndarray):
        if len(xy) > 0:
            return xy/self.scale - np.array([self.x_offset, self.y_offset])
        return np.array([])

    def world_to_img(self, xy):
        if len(xy) > 0:
            return (xy+np.array([self.x_offset, self.y_offset]))*self.scale
        return np.array([])
    
    def lane_number(self, db, X: np.ndarray):
        # change the function name
        '''
            combine clusters which are close to each other
        '''
        core_samples_mask = np.zeros_like(db.labels, dtype=bool)
        core_samples_mask[db.core_sample_indices] = True # All core points are true, else false
        unique_labels = set(db.labels) # gets all the labels
        # label -1 which is for noisy points
        unique_labels.remove(-1)
        new_labels = {}

        # centroids = np.zeros((len(unique_labels), 2))
        centroids = {} # dict of centroids of all lanes 
        for k in unique_labels:
            if k == -1:
                continue
            class_member_mask = (unique_labels == k) # if label is k
            xy_core = X[class_member_mask & core_samples_mask] # if label is k and it is a core point
            xy_non_core = X[class_member_mask & ~core_samples_mask] # if label is k and it is not a core point
            xy = np.concatenate([xy_core, xy_non_core]) # all points of label k
            new_labels[k] = xy
            centroids[k] = [0, 0]
            centroids[k][0] = sum(xy[:, 0])/xy.shape[0]
            centroids[k][1] = sum(xy[:, 1])/xy.shape[0]

        for i in range(len(unique_labels)):
            if unique_labels[i] not in centroids.keys:
                continue
            for j in range(i+1, len(unique_labels)):
                if unique_labels[j] not in centroids.keys:
                    continue
                distance = np.sqrt(
                    (centroids[unique_labels[i]][0]-centroids[unique_labels[j]][0])**2 + (centroids[unique_labels[j]][1]-centroids[unique_labels[j]][1])**2)
                if distance < 1:
                    new_labels[i] = np.concatenate([new_labels[i], new_labels[j]])
                    # update centroids -> i
                    n1 = len(unique_labels[i]) # no. of points in the ith cluster
                    n2 = len(unique_labels[j]) # no. of points in the jth cluster        
                    centroids[unique_labels[i]][0] = (centroids[unique_labels[i]][0]*n1 + centroids[unique_labels[j]][0]*n2)/ (n1+n2)
                    centroids[unique_labels[i]][1] = (centroids[unique_labels[i]][1]*n1 + centroids[unique_labels[j]][1]*n2)/ (n1+n2)

                    # delete centroids -> j
                    del centroids[unique_labels[j]]
                    del new_labels[unique_labels[j]]
        
        lane_coeff = [self.left_lane.coeff[2], self.mid_lane.coeff[2], self.right_lane.coeff[2]]
        
        for i in range(len(unique_labels)):
            # find the lane polynomial
            cluster_coeff = self.poly_find(new_labels[i])
            

        return new_labels


lanes = Lanes()

def image_callback(msg):
    global coeff
    global lanes

    try:
        img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    # blank_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # np.findnonzero , np.find(img > 128)
    indexes_points = [] # all the lane points
    for index, element in np.ndenumerate(img):
        if element > 128:
            # check if it is white
            indexes_points.append(tuple([index[0], index[1]])) # adds image pixel coordinates to the list

    indexes_points = np.array(indexes_points)
    # This is for the normal indices which we get at the correct time stamp
    db = DBSCAN(eps=7, min_samples=25, algorithm='auto').fit(indexes_points) # clusters lane points
    # This will give a dictionary containing unique labels and points(combines clusters too)
    # this is passed to sort_lanes

    labels = lanes.lane_number(db, indexes_points)
    final_labels = dummy_lane.sort_lanes()
    # from all the lanes find the closest 3 lanes to prev lanes
    # Index out of bounds??
    abc_left = left_lane.poly_find(final_labels[list(final_labels)[0]])
    abc_mid = mid_lane.poly_find(final_labels[list(final_labels)[1]])
    abc_right = right_lane.poly_find(final_labels[list(final_labels)[2]])

    abc_left = left_lane.prediction(
        abc_left, final_labels[list(final_labels)[0]])
    abc_mid = mid_lane.prediction(abc_mid, final_labels[list(final_labels)[1]])
    abc_right = right_lane.prediction(
        abc_right, final_labels[list(final_labels)[2]])
    coeff = [list(abc_left), list(abc_mid), list(abc_right)]
    dummy_lane.poly_viz()


def main():
    image_topic = "top_view"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
