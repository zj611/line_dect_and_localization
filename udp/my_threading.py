# -*- coding: utf-8 -*-
import cv2
import socket
import numpy as np
import threading
from ImgStack import Stack

def capturing_thread(video_path, frame_buffer, lock):
    print("capture_thread start")
    cap = cv2.VideoCapture(video_path)
    while (True):
        success, frame = cap.read()
        while not success and frame is None:
            print ("no image")
            success,frame=cap.read()  #获取视频帧
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
 
def sending_thread(frame_buffer, lock, HOST = '192.168.31.***', PORT = 8888):
    num = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        if frame_buffer.size() > 2:
            num = num+1
            #print('\n-----------No.',num,'------------')
            lock.acquire()
            frame = frame_buffer.pop()
            lock.release()
            frame = cv2.resize(frame,(320,240))
            img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,50])[1]
            data_encode = np.array(img_encode)
            data = data_encode.tostring()

            # 发送数据:
            s.sendto(data, (HOST,PORT))
            print ("succeed")
    s.close()
