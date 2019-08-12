# -*- coding: utf-8 -*-
import sys
import threading
import cv2
from ImgStack import Stack
from my_threading import capturing_thread, sending_thread

if __name__ == '__main__':
    # raspi 读取摄像头，默认为video0
    path = 0
    frame_buffer = Stack(3)
    
    host='192.168.31.***'
    port=9999
    
    # 线程1 读取视频图像送入队列中
    t1 = threading.Thread(target = capturing_thread, args=(path, frame_buffer, threading.RLock()))
    t1.start()
    # 线程2 发送视频
    t2 = threading.Thread(target = sending_thread, args=(frame_buffer, threading.RLock(), host, port))
    t2.start()

