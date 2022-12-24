#! /usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Instantiate CvBridge
bridge = CvBridge()
pub = rospy.Publisher("check1", Image, queue_size=10)


def model_f(x, a, b, c):
    return a*(x-b)**2+c


def image_callback(msg):
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "mono8")
    except CvBridgeError as e:
        print(e)

    # the cv_img is a 2d array

    cv2_img_transpose = np.transpose(cv2_img)

    sum_list = []

    for col in cv2_img_transpose:
        sum_list.append(np.sum(col)/255)

    x = list(range(0, cv2_img.shape[1]))

    # plt.plot(x, sum_list)
    # plt.xlabel('x - axis')
    # plt.ylabel('y - axis')
    # plt.title('histogram of pixel values')
    # plt.show()

    # I found out the approximate regions where the graph should lie
    # for left lane column numbers are 191-210
    # middle lane column numbers are 250 to 265
    # for right lanes column numbers are 307 to 358
    first_lane_x = x[191:211]
    middle_lane_x = x[250:266]
    last_lane_x = x[307:358]

    lane1 = []
    lane2 = []
    lane3 = []
    # now i need the u , v coordinate of the points in the respective regions
    for index, element in np.ndenumerate(cv2_img):
        index_l = list(index)
        if element == 255:
            if 170 <= index[1] < 220:
                lane1.append(index_l)
            if 250 <= index[1] < 270:
                lane2.append(index_l)
                #pub.publish(bridge.cv2_to_imgmsg(image, "mono8"))
            if 307 <= index[1] < 358:
                lane3.append(index_l)
                # print(index[0])

    lane1 = np.array(lane1)
    lane2 = np.array(lane2)
    lane3 = np.array(lane3)

    blank_img = np.zeros(
        (cv2_img.shape[0], cv2_img.shape[1], 3), dtype=np.uint8)

    if lane2.size > 0:
        popt, pcov = curve_fit(model_f, lane2[:, 1], lane2[:, 0], [-3, -7, 1])

        a_opt, b_opt, c_opt = popt
        #x_model = np.linspace(255, 270, 15).tolist()
        x_model = range(250, 270)
        y_model = model_f(x_model, a_opt, b_opt, c_opt).tolist()
        for i in range(0, len(x_model)):
            blank_img = cv2.circle(
                blank_img, tuple([int(x_model[i]), int(y_model[i])]), radius=5, color=(0, 0, 255), thickness=-1)

    if lane1.size > 0:
        popt, pcov = curve_fit(model_f, lane1[:, 1], lane1[:, 0], [-3, -7, 5])

        a_opt, b_opt, c_opt = popt
        #x_model = np.linspace(180, 211, 31).tolist()
        x_model = range(150, 200)
        y_model = model_f(x_model, a_opt, b_opt, c_opt).tolist()
        for i in range(0, len(x_model)):
            blank_img = cv2.circle(
                blank_img, tuple([int(x_model[i]), int(y_model[i])]), radius=5, color=(255, 0, 0), thickness=-1)
        print(popt)

    if lane3.size > 0:
        popt3, pcov3 = curve_fit(
            model_f, lane3[:, 1], lane3[:, 0], [-3, -7, 1])

        a_opt3, b_opt3, c_opt3 = popt3
        #x_model = np.linspace(307, 358, 50).tolist()
        x_model = range(300, 358)
        y_model = model_f(x_model, a_opt3, b_opt3, c_opt3).tolist()
        for i in range(0, len(x_model)):
            blank_img = cv2.circle(
                blank_img, tuple([int(x_model[i]), int(y_model[i])]), radius=5, color=(0, 255, 255), thickness=-1)
    #plt.plot(first_lane_x, lane1)
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    #plt.title('Whether the points are visible in matplotlib')
    # plt.show()
    pub.publish(bridge.cv2_to_imgmsg(blank_img, "passthrough"))


def main():
    rospy.init_node('polyfit')
    # Define your image topic
    image_topic = "/cv_pub"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()


if __name__ == '__main__':
    main()
