#!/usr/bin/python3

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from socket import *
import cv2
import numpy as np
#print(len(sys.argv))
if (len(sys.argv)!= 2):
    print("----------------usage guide----------------------")
    print("Usage: python VideoCapture.py your-video-name")
    print("Message: --space-- is for stopping!")
    print("Message: --esc-- is for ending and saving video!")
    print("-------------------------------------------------")
    exit()

max_count_frames = 10000000
#video = "http://192.168.31.144:8080/video"

 
sz = (640,480)
 
fps = 20
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
#fourcc = cv2.VideoWriter_fourcc(*'mpeg')
 
video = cv2.VideoWriter()
name = str(sys.argv[1])+'.mp4'
#name = 'output1111.mp4'
video.open(name,fourcc,fps,sz,True)
host_path = "192.168.31.228"
cnt = 0

cv2.namedWindow("video")
# while (1):
#     print(sz)
#     tem = cv2.waitKey(2)
#     #print(tem)
#     if (tem == 32):#space == 32
#         print("-------------录制暂停----------------")
#         cv2.waitKey(0)
#     if (tem == 27):#esc == 27
#         print("-------------录制结束----------------")
#         break

#     if (cnt > max_count_frames):
#         print("max_count_frames is coming!")
#         break
    
    
    #b_value, frame = cap.read()

s = socket(AF_INET, SOCK_DGRAM)
# 绑定端口:
s.bind((host_path, 9999))
while True:
# 接收数据:
    cnt += 1
    print("Frame id: "+str(cnt))
    data, addr = s.recvfrom(400000)
    # 解码
    nparr = np.fromstring(data, np.uint8)
    # 解码成图片numpy
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.resize(img_decode, (640,480))
    cv2.putText(frame, "Frame id: "+str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('video',frame)
    video.write(frame)
    cv2.waitKey(1)
 
video.release()
cap.release()
cv2.destroyAllWindows()

