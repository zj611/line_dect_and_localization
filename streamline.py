import cv2
import numpy as np

def elbow_bend(frame):

    xmin = 0
    xmax = 640

    center = (xmax - xmin) / 2
    ymin = 65
    ymax = 175
    x_bottom = xmax - xmin
    b_d = 0
    area = 0
    horizontal_line = 0
    dotted_line = 0
    bfindzhijiao = False
    

    # ---------------判断条件----------------
    y_d_u = 5  # 识别的y之间的差值最大值 10
    y_d_l = 0  # 识别的y之间的差值最小值 1
    x_d_u = 150  # 识别的x之间的差值最大值 100
    x_d_l = 15  # 识别的x之间的差值最小值 5
    b_d_l = 225  # 识别的横线到底部之间的差值最小值
    b_d_u = 250  # 识别的横线到底部之间的差值最大值
    area_u = 2000  # 识别的轮廓区域最大值
    area_l = 900  # 识别的轮廓区域最小值
    # --------------判断条件----------------------
    
    show_img = frame.copy()
    cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    img1 = frame[ymin:ymax, xmin:xmax]
    img2 = img1
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area < area_u) and (area > area_l):
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
            if (abs(k) <= 0.08) and (abs(x_d) > x_d_l) and (abs(x_d) < x_d_u) and (abs(y_d) > y_d_l) and (abs(y_d) < y_d_u) and (x1 < center) and (x1 > center - 200) and (x2 > (center - 200)) and (x2 < (center + 80)):
                    horizontal_line = 1
                    kb = k
                    print('kb:',kb)
                    bb = y2 - kb * x2
                    b_d = (y2 + y1) / 2 + ymax
                    if horizontal_line == 1:
                        cv2.line(show_img, (x_bottom, int(kb * x_bottom + bb)), (0, int(bb)), (200, 125, 5), 2)
    #cv2.imshow('elbow',show_img)
    
    if horizontal_line == 1 and dotted_line == 1 and (b_d > b_d_l) and (b_d < b_d_u):
        bfindzhijiao = True
    return bfindzhijiao