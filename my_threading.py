
import threading
import _thread
import cv2
import numpy as np

import time
from socket import *

from func_line_keeping import lane_keeping

import yaml

EV3_addr = ("192.168.31.154",8888)
EV3_socket = socket(AF_INET,SOCK_DGRAM)
cap = cv2.VideoCapture(0)

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
            for i in range(0,self.size() - self.stack_size + 1):
                self.items.remove(self.items[0])
        self.items.append(item)

def capturing_thread(video_path, frame_buffer, lock):
    print("capture_thread start")
    cap = cv2.VideoCapture(video_path)
  
    while (True):
        _, frame = cap.read()
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
        # key = cv2.waitKey(2)
        # if(key == 27):
        #     break
    
            

        #print('size:',frame_buffer.size())
        # key = cv2.waitKey(2)#-----------------------------ERROR!!!!!!!!--------


K = np.array([[443.46791052598115, 0.0, 326.3259540224434], [0.0, 442.9083722265219, 223.37747344757463], [0.0, 0.0, 1.0]],dtype = "float32")
distCoeffs = np.array([-0.38235254174930167, 0.12374691224814019, 0.0004041714187102742, -0.00228754147907862],dtype = "float32")

xmin = 20   
xmax = 600 
center = (xmax + xmin) / 2
x_center = center
ymin = 170
ymax = 450





def processing_thread(frame_buffer, lock):
    print("detect_thread start")
    # print("detect_thread frame_buffer size is", frame_buffer.size())
    flag_t = 0 #计算一次透视变换矩阵的标志位
    steering_memory = 0
    cmd_ = 0
    num = 0

    conf = yaml.load(open('config.yaml','r',encoding='utf-8'))

    # f = open('config.yaml','r',encoding='utf-8')
    # cont = f.read()
    # conf = yaml.load(cont)

    global bais,warning_flag,steering_memory,delta_value
    steering_memory = 0 
    lane_memory = conf['lane_memory']  #初始不应该在弯道等识别不了直线的状态，否则，应该给定初始车道位置

    warning_flag = 0
    die_area = conf['die_area']
    delta_value = conf['delta_value']

    t0 = conf['t0']
    speed_value = conf['speed_value']
    cv2.namedWindow("perspective1",cv2.WINDOW_NORMAL)
    cv2.namedWindow("raw img",cv2.WINDOW_NORMAL)
   
    # #----------------------------For actual testing-----------------------------------------
    # while True:
    #     if frame_buffer.size() > 2:
    #         num = num+1
    #         print('\n-----------No.',num,'------------')
    #         lock.acquire()
    #         frame = frame_buffer.pop()
    #         lock.release()
    #         #print(frame.shape)
    
    # #-----------------------------For video dataset------------------------------------------
    video = conf['path']
    cap = cv2.VideoCapture(video)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('size:',size)
    
    num = 0
    while (cap.isOpened()):
        _,s = cap.read()
        if s is not None:
            
            num = num+1
            print('\n-----------No.',num,'------------')
            frame = s.copy()
 
#------------------------------------------------------------------------------------------
    
            s = cv2.undistort(frame,K,distCoeffs)
     
            img0 = s.copy()
            cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)  
            # cv2.circle(img0, (170,100), 5, (0, 0, 255), -1) #左上
            # cv2.circle(img0, (100,300), 5, (0, 0, 255), -1) #左下
            # cv2.circle(img0, (430,100), 5, (0, 0, 255), -1) #右上
            # cv2.circle(img0, (560,300), 5, (0, 0, 255), -1) #右下

            img_roi = s[ymin:ymax,xmin:xmax]
 
            time_start = float(time.time())

 #--------------------------------车道保持程序----------------------------------------------------
            cmd_,current_lane,out_img1 = lane_keeping(img_roi,flag_t,steering_memory,lane_memory)
            lane_memory = current_lane
            steering_memory = cmd_
            #------------------     
            time_cost = (float(time.time()) - time_start) * 1000
            print("\nprocessing time: %.5f ms" % time_cost)

#-----------------------------------------------------------------------------------------------


            direction0 = 'left'
            direction1 = 'right'
            direction2 = 'no steering'

            print('\n')
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

            if(warning_flag == 1):
                cv2.putText(out_img1, text_warning, (40,200), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 255), 2,cv2.LINE_AA)
            
            cv2.imshow('raw img',img0)
            cv2.imshow('perspective1',out_img1)
            
            print('*turn       = '+direction)
            print("*steering   = ",cmd_)
            print("*command    = ",command)

            EV3_socket.sendto(command.encode(),EV3_addr)

            steering_memory = cmd_

            key = cv2.waitKey(t0)

            if key == 27:
            #esc键退出
                print("-----------程序结束--------------")
                filename = "amap.jpg"
                cv2.imwrite(filename, s)
                cap.release()
                cv2.destroyAllWindows()
                break
            if (key == 32):#space == 32
                print("-------------暂停----------------")
                cv2.waitKey(0)