# coding=utf-8
import threading
import _thread
import cv2  
import numpy as np

import time
from time import sleep
from socket import *
from function.func_line_keeping import lane_keeping

from function.rec import rec_dect
from function.stop_line import detect_line
from function.stop_line1 import detect_line1

import sys

from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import yaml

EV3_addr = ("192.168.31.154",8888)
EV3_socket = socket(AF_INET,SOCK_DGRAM)

class Stack:
 
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size
 
    def is_empty(self):
        return len(self.items) == 0
 
    def pop(self):
        return self.items.pop()
 
    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]
 
    def size(self):
        return len(self.items)
 
    def push(self, item):
        
        if self.size() >= self.stack_size:
            for i in range(0,(self.size() - self.stack_size + 1)):
                self.items.remove(self.items[0])
        self.items.append(item)
        

def signal(img,signal):
        out_imgy=np.copy(img)

        hsv = cv2.cvtColor(out_imgy, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR
        if signal < 2 or signal > 9:
            lower_yellow = np.array([25, 0, 200])
            upper_yellow = np.array([40, 255, 255])
        elif signal >= 2 and signal <= 5:
            lower_yellow = np.array([20, 170, 0])
            upper_yellow = np.array([50, 255, 220])
        elif signal >= 6 and signal <= 9:
            lower_yellow = np.array([110, 50, 0])
            upper_yellow = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        if signal >= 6 and signal <= 9:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.erode(mask, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            signal_out = 1
        else:
            signal_out = 0


        # cv2.putText(mask, str(signal_in), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(mask, str(signal_out), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('10',mask)
        return signal,signal_out

def bluehinder(s,sig):
    out_imgy1 = np.copy(s)
    pts1 = np.float32([[163, 1], [416, 1], [49, 479], [518, 479]])
    ab = 130
    ac = 0
    al = 500
    ak = 200
    pts2 = np.float32([[ab, ac], [ab + ak, ac], [ab, al + ac], [ab + ak, al + ac]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out_imgy1 = cv2.warpPerspective(out_imgy1, M, (510, 500))

    hsv = cv2.cvtColor(out_imgy1, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 0])
    upper_blue = np.array([140, 255, 255])
    blue_line = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_line = cv2.erode(blue_line, kernel)
    contours_blue, hierarchy = cv2.findContours(blue_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    ssig = 0
    if len(contours_blue) > 0:
        ssig = 1

    white_imgy = np.copy(s)
    white_hsv = cv2.cvtColor(white_imgy, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(white_hsv, lower_white, upper_white)
    white_mask_1 = 255 - white_mask
    white_imgy = cv2.bitwise_and(white_imgy, white_imgy, mask=white_mask_1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    white_mask = cv2.erode(white_mask, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    white_mask = cv2.dilate(white_mask, kernel)
    contours_white, hierarchw = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #white_imgy = cv2.drawContours(white_imgy, contours_white, -1, (96, 96, 96), cv2.FILLED)

    if sig == 1: 
        lf = 55
    if sig == 2:
        lg = 31  
    if sig == 3: 
        lf = 26
    if sig == 4:
        lg = 35  

    if len(contours_white) != 0:

        for cnt in contours_white:
            if sig == 1:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] - lf
            if sig == 2:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] + lg
            if sig == 3:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] - lf
            if sig == 4:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] + lg
    out_imgy = cv2.drawContours(white_imgy, contours_white, -1, (255, 255, 255), cv2.FILLED)
    return (out_imgy, ssig)

def singlebridge(s,sig,sig_bridge):
    out_imgy1 = np.copy(s[100:480,0:640])
    pts1 = np.float32([[163, 1], [416, 1], [49, 479], [518, 479]])
    ab = 130
    ac = 0
    al = 500
    ak = 200
    pts2 = np.float32([[ab, ac], [ab + ak, ac], [ab, al + ac], [ab + ak, al + ac]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out_imgy1 = cv2.warpPerspective(out_imgy1, M, (510, 500))
    #cv2.imshow('3',out_imgy1)

    kernel_size = 5
    hsv = cv2.cvtColor(out_imgy1, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR
    lower_yellow = np.array([25, 0, 190])
    upper_yellow = np.array([40, 255, 255])
    yellow_line = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_line = cv2.GaussianBlur(yellow_line, (kernel_size, kernel_size), 1.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    yellow_line = cv2.erode(yellow_line, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 30))
    yellow_line = cv2.dilate(yellow_line, kernel)
    contours_yellow, hierarchy = cv2.findContours(yellow_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #out_imgy = cv2.drawContours(out_imgy, contours_yellow, -1, (255, 0, 0), 3)
    ymax = -1
    xmax = -1
    ll = 0
    #area = -1
    ssig = 0
    if len(contours_yellow) > 0:
        ssig = 1
        # cnt = contours_yellow[0]
        # for po in cnt:
        #     if po[0,1] > ymax:
        #         ymax = po[0,1]
        #     if po[0,0] > xmax:
        #         xmax = po[0,0]

        #area = cv2.contourArea(cnt)

    white_imgy = np.copy(s)
    white_hsv = cv2.cvtColor(white_imgy, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(white_hsv, lower_white, upper_white)
    white_mask_1 = 255 - white_mask
    white_imgy = cv2.bitwise_and(white_imgy, white_imgy, mask=white_mask_1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    white_mask = cv2.erode(white_mask, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    white_mask = cv2.dilate(white_mask, kernel)
    contours_white, hierarchw = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #white_imgy = cv2.drawContours(white_imgy, contours_white, -1, (96, 96, 96), cv2.FILLED)

    #if ymax < 490 and ymax > 0:
    # if ymax == -1:
    lg = 52#73-90
    lf = 95
    if sig == 1:
        lg = 37

    if sig_bridge == 0:
        lg = 20
        lf = 40
    # if ymax == -1:
    #     ll = -lg
    #     lg = -lf
    #     lf = ll
    # else:
    #     lg = 55
    #     lf = 75

    if len(contours_white) != 0:

        for cnt in contours_white:
            if sig == 1:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] + lg
            if sig == 2:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] - lf
            if sig == 3:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] + lg
    out_imgy = cv2.drawContours(white_imgy, contours_white, -1, (255, 255, 255), cv2.FILLED)
    #cv2.imshow('2', white_mask)
    # cv2.putText(out_imgy, 'ymax:'+str(ymax), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(out_imgy, 'xmax:'+str(xmax), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    return (out_imgy, ssig)


class capture_thread2(QtCore.QThread):
    def __init__(self,path,frame_buffer,lock):
        super(capture_thread2, self).__init__()
        self.path = path
        self.frame_buffer = frame_buffer
        self.lock = lock

    def __del__(self):
        self.wait()

    def run(self):
        print("capture_thread2 start")
        s = socket(AF_INET, SOCK_DGRAM)
        # 绑定端口:
        s.bind((self.path, 9999))
        while True:
        # 接收数据:
            data, addr = s.recvfrom(400000)
            # 解码
            nparr = np.fromstring(data, np.uint8)
            # 解码成图片numpy
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_decode = cv2.resize(img_decode, (640,480))
            self.lock.lock()
            self.frame_buffer.push(img_decode)
            self.lock.unlock()

class capture_thread0(QtCore.QThread):
    # python3,pyqt5与之前的版本有些不一样
    #  通过类成员对象定义信号对象
        def __init__(self,path,frame_buffer,lock):
            super(capture_thread0, self).__init__()#继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
            self.path = path
            self.frame_buffer = frame_buffer
            self.lock = lock

        def __del__(self):
            self.wait()
    
        def run(self):
            print("capture_thread0 start")
            print(self.path)
            cap = cv2.VideoCapture(self.path)
            while (cap.isOpened):
                _, frame = cap.read()
                self.lock.lock()
                self.frame_buffer.push(frame)
                self.lock.unlock()
            exit()

K = np.array([[443.46791052598115, 0.0, 326.3259540224434], [0.0, 442.9083722265219, 223.37747344757463], [0.0, 0.0, 1.0]],dtype = "float32")
distCoeffs = np.array([-0.38235254174930167, 0.12374691224814019, 0.0004041714187102742, -0.00228754147907862],dtype = "float32")

xmin = 20   
xmax = 600 
center = (xmax + xmin) / 2
x_center = center
ymin = 170
ymax = 450



# 继承QThread
class process_thread0(QtCore.QThread):
    _signal = pyqtSignal(str,str)
    _signal_img = pyqtSignal(QImage)
 
    def __init__(self, frame_buffer,lock):
        super(process_thread0, self).__init__()
        self.frame_buffer = frame_buffer
        self.lock = lock
 
    def __del__(self):
        self.wait()
 
    def run(self):
        print("process_thread start")
        flag_t = 0 #计算一次透视变换矩阵的标志位

        cmd_ = 0
        num = 0
        num_rele = 0
        num_rec = 0  #直角弯提速
        change_cmd = 0  #0-no change  1-change to left  2-change to right
        current_lane = 1
        yellow_is_open = 0  #0-close 1-open

        conf = yaml.load(open('config.yaml','r',encoding='utf-8'), Loader=yaml.FullLoader)
        
        global bais,warning_flag,steering_memory,delta_value
        steering_memory = 0 
        lane_memory = conf['lane_memory']  #初始不应该在弯道等识别不了直线的状态，否则，应该给定初始车道位置
        
        warning_flag = 0
        die_area = conf['die_area']
        delta_value = conf['delta_value']
        
        t0 = conf['t0']
        speed_value = conf['speed_value']
        
        # cv2.namedWindow("perspective1",cv2.WINDOW_NORMAL)
        # #cv2.namedWindow("raw img",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
   
        #-----video writer0------
        fps = 10
        sz = (510,500)
        #fourcc =  cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        #fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
        #fourcc = cv2.VideoWriter_fourcc(*'mpeg')
        video_wirte = cv2.VideoWriter()
        name = 'mission_record.mp4'
        video_wirte.open(name,fourcc,fps,sz,True)
        #-----video writer1------
        fps = 10
        sz = (640,480)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_ini = cv2.VideoWriter()
        name = 'video_ini.mp4'
        video_ini.open(name,fourcc,fps,sz,True)
        #----------file write---
        fo = open("cammand_rec.txt", "w")
        #------------------------

        
        flag_rec = False
        flag_stop = False
        yellow_is_open = 0 #yellow is closed
        #######################################################
        #从出发开始    program = -1  sign = 1
        #调试蓝色障碍  program = 9  sign = 6
        #调试终止停车线  program = 6  sign = 1
        #调试单边桥     program = -1   sign = 7
        #调试限宽門     program = 9   sign = 2
        
        program = -1
        sign = 1
        #######################################################

        num_0 = 0
        num_1 = 0
        signn = 1
        tes = 0
        one_change_cmd = 0
        
        sig_bridge = 1
        loc_bridge = 1

        sig_blue = 1
        loc_blue = 1

    ##-------blue_barrier-----------
        #global barrier,n_obstacle
        barrier = 0   #障碍物
        n_obstacle = 0#障碍物个数
        flag_blue = False
    #------------------------------

        # #----------------------------测试用，整段注释-------------------------------------------
        while True:
            if (self.frame_buffer.size() > 2):
                time_start = float(time.time())
                num = num+1
                print('\n-----------No.',num,'------------')
                self.lock.lock()
                frame = self.frame_buffer.pop()
                self.lock.unlock()
                #print(frame.shape)
        
        # #-----------------------------调试用，整段注释------------------------------------------
        # video = conf['path']
        
        # cap = cv2.VideoCapture(video)
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print('size:',size)
        # num = 0
        
        # while (cap.isOpened()):
        #     _,s = cap.read()
        #     time_start = float(time.time())
        #     if s is not None:
        #         num = num+1
        #         print('\n-----------No.',num,'------------')
        #         frame = s.copy()
    
    #------------------------------------------------------------------------------------------
          
                s = cv2.undistort(frame,K,distCoeffs)
                img0 = s.copy()
                #起步稳定后提速
                if(num == 5):
                    speed_value = 90
   
    #--------------------------------------高层决策---------------------------------------------
                ss = s[0:img0.shape[0]-50, 0:img0.shape[1]]
                _,s_out = signal(ss,sign)
                if s_out == 0:
                    num_0 = num_0 + 1
                    num_1 = 0
                if s_out == 1:
                    num_1 = num_1 + 1
                    num_0 = 0

                    litin = 2
                    litout = 2
                
                if sign == 6:
                    litin =  2
                    litout = 5
                if sign == 7:
                    litin =  2 
                    litout = 3 #5
                
                if sign == 8:
                    litin =  2
                    litout = 5 #7
                
                if sign == 9:
                    litin = 2 #3
                    litout= 5
     
                if sign == 1:
                    litin = 3
                    litout = 3

                if sign == 2:
                    litin = 2
                    litout = 2

                if sign == 5:
                    litin = 5
                    litout = 0
                if sign == 10:
                    litin = 3
                    litout = 1
                if sign > 10:
                    litin = 0
                    litout = 1

                if num_1 > litin:
                    tes = 1
                if num_0 > litout:
                    tes = 0

                if tes == 1 and signn == 1:
                    signn = 0
                if signn == 0 and tes == 0:
                    signn = 1
                    sign = sign + 1

                cv2.putText(img0, str(num_0), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img0, str(num_1), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('00',img0)

                if sign == 1 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'start', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                if sign == 1 and tes == 1:
                    cv2.putText(img0, 'deceleration strip', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                    program = 0 #直角弯
                    speed_value = 50
                    
                if sign == 2 and tes == 0 and signn == 1 and program == 0:
                    cv2.putText(img0, 'start L and yellow open' , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    yellow_is_open = 1
                    speed_value = 70
                    

                if sign == 2 and tes == 1:
                    cv2.putText(img0, 'door_1_in and yellow close', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    yellow_is_open = 0
                    speed_value = 50
  

                if sign == 3 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'door_1_out and turn left', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    change_cmd = 1   #左车道
                                   

                if sign == 3 and tes == 1:
                    cv2.putText(img0, 'door_2_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #one_change_cmd = 1
                if sign == 4 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'door_2_out and turn right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    change_cmd = 2
                                 

                if sign == 4 and tes == 1:
                    cv2.putText(img0, 'door_3_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
               
                if sign == 5 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'door_3_out and turn left', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    change_cmd = 1
             
                if sign == 5 and tes == 1:
                    cv2.putText(img0, 'door_4_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
       
                if sign == 6 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'door_4_out and force right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    change_cmd = 2
                    speed_value = 45
             
                if sign == 6 and tes == 1 and program == 9:
                    cv2.putText(img0, 'hinder_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    program = 22 #蓝色障碍物
                    speed_value = 30
                
                if sign == 7 and tes == 0 and signn == 1 and program == 22:
                    change_cmd = 1
                    program = 23
                    speed_value = 50

                if sign == 7 and tes == 1:
                    speed_value = 40

                if sign == 8 and tes == 0 and signn == 1 and program == 23:
                    change_cmd = 2
                    program = 24
                    speed_value = 50
                
                if sign == 8 and tes == 1:
                    speed_value = 40
                
                if sign == 9 and tes == 0 and signn == 1 and program == 24:
                    change_cmd = 1
                    program = 25
                    speed_value = 50
                
                if sign == 9 and tes == 1:
                    speed_value = 40

                if sign == 10 and tes == 0 and signn == 1:
                    program = 3
                #if program == 33:
                    cv2.putText(img0, 'hinder_out and turn right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    change_cmd = 2  #蓝色障碍物出，保持右道
                    speed_value = 45
                    num_rele +=1
                    if(num_rele > 20):
                        speed_value = 65
                    if(num_rele > 160 ):
                        speed_value = 60
                    #program = -1
                    

                if sign == 10 and tes == 1:
                    cv2.putText(img0, 'singlebridge_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)
                    program = 4
                    speed_value = 40
                if sign == 13 and tes == 0 and signn == 1:
                    cv2.putText(img0, 'singlebridge_out', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)
                    speed_value = 60
                    program = 6#准备检测任务结束
   
    #--------------------------------任务调度----------------------------------------------------
                if(program == 0):#直角弯
                    flag_rec = rec_dect(s.copy())
                    if flag_rec:
                        print('**************************************************')
                        command = "m1180180"
                        EV3_socket.sendto(command.encode(),EV3_addr)
                        sleep(1.9)
                                  
                        speed_value = 40
                        change_cmd = 2
                        program = 1
                        continue
  
                if(program == 1):                                
                    num_rec+=1
                    if(num_rec > 15):
                        speed_value = 90
                    flag_stop = detect_line(s.copy())
                    print('flag_stop:',flag_stop)
                    
                    if(flag_stop):
                        command = "m1100100"
                        EV3_socket.sendto(command.encode(),EV3_addr)
                        sleep(3)
                        program = 9
                        speed_value = 65

                if(program == 6):
                    flag_stop = detect_line1(s.copy())
                    print('flag_stop:',flag_stop)
                        
                    if(flag_stop):
                        command = "m1000150"
                        EV3_socket.sendto(command.encode(),EV3_addr)
                        sleep(0.9)
                        command = "m1100100"
                        EV3_socket.sendto(command.encode(),EV3_addr)
                        break

                if(program == 4):
                    outsin = singlebridge(s.copy(),loc_bridge,sig_bridge)
                    if outsin[1]==1:
                        sig_bridge = 1
                    #     if xmaxx > 300:
                    #         loc_bridge = 1
                    #         sig_bridge = 0
                    #     else:
                    #         loc_bridge = 0
                    #         sig_bridge = 0
                    if outsin[1] == 0 and sig_bridge == 1:
                        sig_bridge = 0
                        loc_bridge = loc_bridge + 1
                    # cv2.putText(outsin[0], 'loc_bridge:' + str(loc_bridge), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    # cv2.putText(outsin[0], 'sig_bridge:' + str(sig_bridge), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    # cv2.putText(outsin[0], 'outsin[1]:' + str(outsin[1]), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    # cv2.imshow('1', outsin[0])
                
                    s = outsin[0]
                
                if program >= 22:
                    outsin = bluehinder(s.copy(), sign-5)
                    # if outsin[1]==1:
                    #     sig_blue = 1
                    # if outsin[1] == 0 and sig_blue == 1:
                    #     sig_blue = 0
                    #     loc_blue = loc_blue + 1
                    
                    # cv2.putText(outsin[0], 'loc_blue:' + str(loc_blue), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    s = outsin[0]    
                
                if((change_cmd == 1) & (current_lane == 1) | (change_cmd == 2) & (current_lane == 2)): 
                    change_cmd = 0 
                #-------------车道保持-----------------
                cmd_,current_lane,out_img1,gray = lane_keeping(s,flag_t,steering_memory,lane_memory,change_cmd,yellow_is_open)
                lane_memory = current_lane
                #------------------------------------
                cv2.putText(out_img1, 'Speed: '+str(speed_value), (20,110), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2,cv2.LINE_AA)
                cv2.putText(out_img1, 'No.'+str(num), (5, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
                if(yellow_is_open ==1):
                    cv2.putText(out_img1, 'Yellow: Open', (20,150), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 0, 255), 2,cv2.LINE_AA)
               
                # cv2.putText(out_img1, 'program:'+str(program), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
                # cv2.putText(out_img1, str(sign)+":"+'&'+str(s_out), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
                # cv2.putText(out_img1, 'change_lane:'+str(change_cmd), (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
                # cv2.putText(out_img1, 'flag_stop:'+str(flag_stop), (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
                # cv2.putText(out_img1, 'flag_rec:'+str(flag_rec), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
                
                if(program == 0):
                    if(abs(cmd_)>14):
                        cmd_ = 0#直角弯入弯前保护

                #-------------指令编码-----------------
                direction0 = 'left'
                direction1 = 'right'
                direction2 = 'no steering'
                if (cmd_ == 0):
                    command = "m1" + "10" + str(abs(cmd_))+ "1"+str(speed_value)
                    direction = direction2
                elif (cmd_<=-10):
                    command = "m1"  + "0" + str(abs(cmd_))+ "1"+str(speed_value)
                    direction = direction0
                elif (cmd_>=10):
                    command = "m1"  + "1" + str(abs(cmd_))+ "1"+str(speed_value)
                    direction = direction1
                elif (cmd_>0):
                    command = "m1" + "10" + str(abs(cmd_))+ "1"+str(speed_value)
                    direction = direction1
                else:
                    command = "m1" + "00" + str(abs(cmd_))+ "1"+str(speed_value)
                    direction = direction0
                cv2.putText(out_img1, 'turn :'+direction, (int(out_img1.shape[0]/2),50), cv2.FONT_HERSHEY_SIMPLEX , 1, (48, 255, 255), 2,cv2.LINE_AA)
                cv2.putText(out_img1, 'value:'+str(cmd_), (int(out_img1.shape[0]/2),80), cv2.FONT_HERSHEY_SIMPLEX , 1, (48, 255, 255), 2,cv2.LINE_AA)

                # cv2.imshow('raw img',img0)
                # cv2.imshow('perspective1',out_img1)
                # cv2.imshow('gray',gray)

                
                print('*turn       = '+direction)
                print("*steering   = ",cmd_)
                print("*command    = ",command)
                #-----------------------
                # filename1 = "./pic/raw/raw_"+str(num)+".jpg"
                # cv2.imwrite(filename1, frame)
                # filename2 = "./pic/out/outimg_"+str(num)+".jpg"
                # cv2.imwrite(filename2, out_img1)

                # fo.write(str(num)+' '+str(cmd_)+' '+str(speed_value)+'\n')
                # video_ini.write(frame)
                # video_wirte.write(out_img1) 
                #-----------------------
                EV3_socket.sendto(command.encode(),EV3_addr)
                steering_memory = cmd_
            

                time_cost = (float(time.time()) - time_start) * 1000
                print("\nprocessing time: %.2f ms" % time_cost)

                #-------------UI数据发送-----------------
                self._signal.emit(str(cmd_),str(speed_value))
                out_img1 = cv2.cvtColor(out_img1,cv2.COLOR_RGB2BGR)
                img = QtGui.QImage(out_img1.data, 
                                    out_img1.shape[1], 
                                    out_img1.shape[0],
                                    out_img1.shape[1]*out_img1.shape[2], 
                                    QtGui.QImage.Format_RGB888)
                self._signal_img.emit(img)
                #---------------------------------------

                key = cv2.waitKey(t0)

                if key == 27: #esc键截取图像
                    filename = './tem_pic/'+str(num)+".jpg"
                    cv2.imwrite(filename, out_img1)
                
                if (key == 32):#space == 32
                    print("-------------暂停----------------")
                    speed_value = 00
                    command = "m1"  + "1" + str(abs(cmd_))+ "1"+str(speed_value)
                    EV3_socket.sendto(command.encode(),EV3_addr)
                    cv2.waitKey(0)

                if(key == 113):
                    change_cmd = 1
                if(key == 119):
                    change_cmd = 2
                
                if(key == 49):
                    speed_value = 10
                if(key == 50):
                    speed_value = 20
                if(key == 51):
                    speed_value = 30
                if(key == 52):
                    speed_value = 40
                if(key == 53):
                    speed_value = 50
                if(key == 54):
                    speed_value = 60
                if(key == 55 ):
                    speed_value = 70
                if(key == 56 ):
                    speed_value = 80
                if(key == 57 ):
                    speed_value = 90
                
                if(key == 96 ):
                    speed_value = 00
                

        cv2.destroyAllWindows()
        print('\n------------------------------------')
        print('Mission is ending! Congraduation!')
        print('------------------------------------\n')
        # video_wirte.close()
        # video_ini.close()
        fo.close()
        exit()

