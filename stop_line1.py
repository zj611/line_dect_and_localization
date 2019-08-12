# coding=utf-8
import cv2
import numpy as np
from socket import *
import time
def detect_line1(show_img):
    xmin = 0
    xmax = 640
    center = (xmax - xmin) / 2
    ymin = 120
    ymax = 140
    bottom = ymax - ymin
    x_bottom = xmax - xmin
    kb = 0.001
    bb = 0.001
    k = 0
    area = 0
    horizontal_line = 0
    dotted_line = 0
    dotted_linel = 0
    num = 0
    ramp = False
    # ---------------判断条件----------------
    y_d_u = 5  # 识别的y之间的差值最大值 10
    y_d_l = 0  # 识别的y之间的差值最小值 1
    x_d_u = 300  # 识别的x之间的差值最大值 100
    x_d_l = 20  # 识别的x之间的差值最小值 5
    area_u = 2750  # 识别的轮廓区域最大值
    area_l = 2000  # 识别的轮廓区域最小值
    # --------------判断条件----------------------
    img2 = show_img[ymin:ymax, xmin:xmax]
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > area_l):
            dotted_linel = 1
        if(area > area_u):
            dotted_line = 1

        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(mask, low_threshold, high_threshold)

        rho = 1.0  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 40  # minimum number of votes (intersections in Hough grid cell) 100
        min_line_length = 10  # minimum number of pixels making up a line 30 20
        max_line_gap = 5  # maximum gap in pixels between connectable line segments 90
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        if lines is not None:
            lines1 = lines[:, 0, :]
            for x1, y1, x2, y2 in lines1[:]:
                x_d = x2 - x1
                y_d = y2 - y1
                if x_d == 0:
                    x_d = 0.0000000001
                k = float(y_d) / x_d
                if k == 0:
                    k = 0.00000000001
                if (abs(k) <= 0.1) and (abs(x_d) > x_d_l) and (abs(x_d) < x_d_u) and (abs(y_d) > y_d_l) and (abs(y_d) < y_d_u) and (x1 < 450) and (x2 < 450) and dotted_linel == 1:
                    horizontal_line = 1
                    kb = k
                    bb = y2 - kb * x2
                    
        if dotted_line == 1 or horizontal_line == 1:
            ramp = True
        else:
            ramp = False
    return ramp
