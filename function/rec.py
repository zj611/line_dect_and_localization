import cv2
import numpy as np

def check_center(center):
    if len(center) == 2:
        dis = abs(center[1][0] - center[0][0])
        if dis > 250:
            return False
        else:
            return True
    elif len(center) == 3:
        dis1 = abs(center[1][0] - center[0][0])
        dis2 = abs(center[2][0] - center[0][0])
        dis3 = abs(center[1][0] - center[2][0])
        dis  = [dis1,dis2,dis3]
        dis = max(dis)
        if dis > 250:
            return True
        else:
            return False

def rec_dect(show_img):

    area = 0
    num = 0
    elbow = False
#---------------判断条件----------------
    ymin = 120
    ymax = ymin + 130
    xmin = 0
    xmax = 640
# --------------判断条件----------------------
   
    cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    img1 = show_img[ymin:ymax, xmin:xmax]
    img2 = img1
    # hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 220])
    # upper_white = np.array([180, 30, 255])
    # mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,230,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("mask",mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = cv2.drawContours(img1, contours, -1, (0, 0, 255), 1)
    validcnt = []
    center = []
    n_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("area=",area)
        if (area > 1100):
            MM = cv2.moments(cnt)
            cx = int(MM['m10']/MM['m00'])
            cy = int(MM['m01']/MM['m00']) 
            center.append([cx,cy])
            validcnt.append(cnt)    
    n_area = len(validcnt)
    if n_area >= 2 and check_center(center):
        elbow = True
        cv2.putText(show_img, 'n_area :' + str(n_area), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(show_img, 'elbow :' + str(elbow), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # cv2.imshow('raw_img', show_img)
    # cv2.imshow('img1', img1)
    return elbow
