
import threading
import _thread
import cv2  
import numpy as np

import time
from time import sleep
from socket import *

from func_line_keeping import lane_keeping
from streamline import elbow_bend

from blue_barrier2 import blue_lane,hinder
from stop_line import detect_line
from stop_line1 import detect_line1



import yaml

EV3_addr = ("192.168.31.154",8888)
EV3_socket = socket(AF_INET,SOCK_DGRAM)
# cap = cv2.VideoCapture(0)

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
        if signal < 2 or signal > 6:
            lower_yellow = np.array([25, 0, 200])
            upper_yellow = np.array([40, 255, 255])
        elif signal >= 2 and signal <= 5:
            lower_yellow = np.array([20, 170, 0])
            upper_yellow = np.array([50, 255, 220])
        elif signal == 6:
            lower_yellow = np.array([110, 50, 0])
            upper_yellow = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.erode(mask, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            signal_out = 1
        else:
            signal_out = 0


        # cv2.putText(mask, str(signal_in), (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(mask, str(signal_out), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('1',mask)
        return signal,signal_out

def singlebridge(s,sig):
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
    #area = -1
    ssig = 0
    if len(contours_yellow) > 0:
        ssig = 1
        cnt = contours_yellow[0]
        for po in cnt:
            if po[0,1] > ymax:
                ymax = po[0,1]
            if po[0,0] > xmax:
                xmax = po[0,0]

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
    if ymax < 490 and ymax > 0:
        lg = 70
        lf = 90
    elif ymax == -1:
        lg = 0
        lf = 0
    else:
        lg = 70
        lf = 90

    if len(contours_white) != 0:
        for cnt in contours_white:
            if sig == 1:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] + lg
            if sig == 0:
                for poi in cnt:
                    poi[0, 0] = poi[0, 0] - lf
    out_imgy = cv2.drawContours(white_imgy, contours_white, -1, (255, 255, 255), cv2.FILLED)
    #cv2.imshow('2', white_mask)
    cv2.putText(out_imgy, 'ymax:'+str(ymax), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(out_imgy, 'xmax:'+str(xmax), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    return (out_imgy, ssig ,xmax)

# host_path 是本机端口
def capture_thread2(host_path, frame_buffer, lock):
    print("capture_thread start")
    s = socket(AF_INET, SOCK_DGRAM)
    # 绑定端口:
    s.bind((host_path, 9999))
    while True:
    # 接收数据:
        data, addr = s.recvfrom(400000)
        # 解码
        nparr = np.fromstring(data, np.uint8)
        # 解码成图片numpy
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_decode = cv2.resize(img_decode, (640,480))
        lock.acquire()
        frame_buffer.push(img_decode)
        #print('frame_buffer.size:',frame_buffer.size())
        
        lock.release()
        # cv2.imshow('uuu',img_decode)
        # cv2.waitKey(1)
        # print("Received image")
        # print(img_decode.shape)


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

def sleep_time(t):
    t = 100000*t
    while(t>0):
        t-=1


def processing_thread(frame_buffer, lock):
    print("detect_thread start")
    # print("detect_thread frame_buffer size is", frame_buffer.size())
    flag_t = 0 #计算一次透视变换矩阵的标志位

    cmd_ = 0
    num = 0
    change_cmd = 0  #0-no change  1-change to left  2-change to right
    current_lane = 1
    yellow_is_open = 0  #0-close 1-open

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
    cv2.namedWindow("gray",cv2.WINDOW_NORMAL)

    #-----video writer------
    fps = 10
    sz = (510,500)
    #fourcc =  cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
    #fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    video_wirte = cv2.VideoWriter()
    #name = 'follow_lane.mp4'
    name = 'mission_record.mp4'
    video_wirte.open(name,fourcc,fps,sz,True)
    #-----------------------
    flag_rec = False
    flag_stop = False
    yellow_is_open = 0 #yellow is closed
    #######################################################
    #从出发开始    program = -1  sign = 1
    #调试蓝色障碍  program = 9  sign = 6
    #调试终止停车线  program = 6  sign = 1
    #调试单边桥     program = 7   sign = -1
    program = -1  
    sign = 1
    #######################################################

    num_0 = 0
    num_1 = 0
    signn = 1
    tes = 0
    one_change_cmd = 0
    # flag_stop_t = 0

    sig_bridge = 1
    loc_bridge = -1

##-------blue_barrier-----------
    #global barrier,n_obstacle
    barrier = 0   #障碍物
    n_obstacle = 0#障碍物个数
    flag_blue = False
#------------------------------

    # #----------------------------测试用，整段注释-------------------------------------------
    while True:
        if (frame_buffer.size() > 2):
            num = num+1
            print('\n-----------No.',num,'------------')
            lock.acquire()
            frame = frame_buffer.pop()
            lock.release()
            #print(frame.shape)
    
    # #-----------------------------调试用，整段注释------------------------------------------
    # video = conf['path']
    # cap = cv2.VideoCapture(video)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print('size:',size)
    # num = 0

    # while (cap.isOpened()):
    #     _,s = cap.read()
    #     if s is not None:
    #         num = num+1
    #         print('\n-----------No.',num,'------------')
    #         frame = s.copy()
 
#------------------------------------------------------------------------------------------
            #cv2.imshow('hhh',frame)
            s = cv2.undistort(frame,K,distCoeffs)
            img0 = s.copy()

#--------------------------------------高层决策---------------------------------------------

            ss = s[0:img0.shape[0]-50, 0:img0.shape[1]]
            _,s_out = signal(ss,sign)
            if s_out == 0:
                num_0 = num_0 + 1
                num_1 = 0
            if s_out == 1:
                num_1 = num_1 + 1
                num_0 = 0

                litin = 3
                litout = 3
            if sign == 6:
                litin =  3
                litout = 100
            if sign == 1:
                litin = 3
                litout = 3

            if sign == 2:
                litin = 2
                litout = 3

            if sign == 5:
                litin = 5
                litout = 0
            if sign == 7:
                litin = 3
                litout = 1
            if sign > 7:
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

            if sign == 1 and tes == 0 and signn == 1:
                cv2.putText(img0, 'start', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
            if sign == 1 and tes == 1:
                cv2.putText(img0, 'deceleration strip', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                program = 0 #直角弯
                speed_value = 30
            if sign == 2 and tes == 0 and signn == 1:
                cv2.putText(img0, 'start L and yellow open' , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                yellow_is_open = 1

            if sign == 2 and tes == 1:
                cv2.putText(img0, 'door_1_in and yellow close', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                yellow_is_open = 0
                speed_value = 30
                # flag_stop = 0
                #one_change_cmd = 1

            if sign == 3 and tes == 0 and signn == 1:# and one_change_cmd == 1:
                cv2.putText(img0, 'door_1_out and turn left', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                change_cmd = 1   #左车道
                #one_change_cmd = 0
                

            if sign == 3 and tes == 1:
                cv2.putText(img0, 'door_2_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #one_change_cmd = 1
            if sign == 4 and tes == 0 and signn == 1:# and one_change_cmd == 1:
                cv2.putText(img0, 'door_2_out and turn right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                change_cmd = 2
                #one_change_cmd = 0
                

            if sign == 4 and tes == 1:
                cv2.putText(img0, 'door_3_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #one_change_cmd = 1
            if sign == 5 and tes == 0 and signn == 1:# and one_change_cmd ==1:
                cv2.putText(img0, 'door_3_out and turn left', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                change_cmd = 1
                #one_change_cmd = 0
                

            if sign == 5 and tes == 1:
                cv2.putText(img0, 'door_4_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #one_change_cmd = 1
            if sign == 6 and tes == 0 and signn == 1:# and one_change_cmd ==1:
                cv2.putText(img0, 'door_4_out and force right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                change_cmd = 2
                speed_value = 40
                #one_change_cmd = 0
                
           
            if sign == 6 and tes == 1 and program == 9:
                cv2.putText(img0, 'hinder_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                program = 2 #蓝色障碍物
            
            # if sign == 7 and tes == 0 and signn == 1:
            if program == 3:
                cv2.putText(img0, 'hinder_out and turn right', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                change_cmd = 2  #蓝色障碍物出，保持右道
                #program = -1
                

            if sign == 7 and tes == 1:
                cv2.putText(img0, 'singlebridge_in', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)
                program = 4
            if sign == 10 and tes == 0 and signn == 1:
                cv2.putText(img0, 'singlebridge_out', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)
                speed_value = 30
                program = 6#准备检测任务结束
#------------------------------------------------------------------------------------------


            # cv2.rectangle(img0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)  
            # cv2.circle(img0, (170,100), 5, (0, 0, 255), -1) #左上
            # cv2.circle(img0, (100,300), 5, (0, 0, 255), -1) #左下
            # cv2.circle(img0, (430,100), 5, (0, 0, 255), -1) #右上
            # cv2.circle(img0, (560,300), 5, (0, 0, 255), -1) #右下

            # img_roi_lane_keeping = s[ymin:ymax,xmin:xmax]
            # img_roi_following_center = s[50:450,20:600]
            # cv2.rectangle(img0, (20, 50), (50, 460), (70, 155, 0), 3)  

            time_start = float(time.time())

 #--------------------------------车道保持程序----------------------------------------------------
            # print('-------------------flag_stop:',flag_stop)
            if(program == 0):#直角弯
                flag_rec = elbow_bend(s.copy())
                if flag_rec:
                    print('**************************************************')
                    command = "m1180180"
                    EV3_socket.sendto(command.encode(),EV3_addr)
                    sleep(1.6)
                    #sleep_time(20)
                    program = 1
                    speed_value = 50
                    change_cmd = 2
                    # flag_stop = 2
                    continue
            
            
            if(program == 1):
                flag_stop = detect_line(s.copy())
                print('flag_stop:',flag_stop)
                
                if(flag_stop):
                    command = "m1100100"
                    EV3_socket.sendto(command.encode(),EV3_addr)
                    sleep(3)
                    program = 9

            if(program == 6):
                flag_stop = detect_line1(s.copy())
                print('flag_stop:',flag_stop)
                      
                if(flag_stop):
                    command = "m1005130"
                    EV3_socket.sendto(command.encode(),EV3_addr)
                    sleep(3.5)
                    command = "m1100100"
                    EV3_socket.sendto(command.encode(),EV3_addr)
                    break

            #blue_barrier = 0
            if(program == 2):#蓝色障碍物
                cx,cy = hinder(s.copy())
                if cx == 255 and cy == 499:
                    barrier = 0
                if barrier == 0:
                    if cy - 500 < -1:
                        barrier = 1
                        n_obstacle += 1
                print('barrier=',barrier)
                print('n_obstacle = ',n_obstacle)
                gather =[cx,cy,barrier,n_obstacle]
                flag_blue = blue_lane(gather)
                
                cv2.waitKey(100)
                if(flag_blue):
                    program = 3
                    speed_value = 40
                   
                    continue
                    
                else:
                    cv2.waitKey(100)
                    continue
            
            if(program == 4):
                outsin = singlebridge(s.copy(),loc_bridge)
                xmaxx = outsin[2]
                if outsin[1]==1 and sig_bridge == 1:
                    if xmaxx > 300:
                        loc_bridge = 1
                        sig_bridge = 0
                    else:
                        loc_bridge = 0
                        sig_bridge = 0
                if outsin[1] == 0 and sig_bridge == 0:
                    sig_bridge = 1
                    loc_bridge = -1
                # cv2.putText(outsin[0], 'loc_bridge:' + str(loc_bridge), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # cv2.putText(outsin[0], 'sig_bridge:' + str(sig_bridge), (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # cv2.putText(outsin[0], 'outsin[1]:' + str(outsin[1]), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                # cv2.imshow('1', outsin[0])
               
                s = outsin[0]


                
             
            if((change_cmd == 1) & (current_lane == 1) | (change_cmd == 2) & (current_lane == 2)): 
                change_cmd = 0 

            cmd_,current_lane,out_img1,gray = lane_keeping(s,flag_t,steering_memory,lane_memory,change_cmd,yellow_is_open)
            lane_memory = current_lane
            #print(out_img1.shape)
            
            # cv2.putText(out_img1, 'program:'+str(program), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
            # cv2.putText(out_img1, str(sign)+":"+'&'+str(s_out), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
            # cv2.putText(out_img1, 'change_lane:'+str(change_cmd), (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
            # cv2.putText(out_img1, 'flag_stop:'+str(flag_stop), (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
            #cv2.putText(out_img1, 'flag_rec:'+str(flag_rec), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
            if(yellow_is_open ==1):
                cv2.putText(out_img1, 'Yellow: Open', (20,150), cv2.FONT_HERSHEY_SIMPLEX , 1.2, (0, 0, 255), 2,cv2.LINE_AA)
        

            #print(out_img1.shape)
            #print(img0.shape)
            video_wirte.write(out_img1)     
            time_cost = (float(time.time()) - time_start) * 1000
            print("\nprocessing time: %.5f ms" % time_cost)
            print('program:',program)

#-----------------------------------------------------------------------------------------------


            direction0 = 'left'
            direction1 = 'right'
            direction2 = 'no steering'

            print('\n')
            print('sign = ',sign)
            if (cmd_ == 0):
                command = "m1" + "10" + str(abs(cmd_))+ "1"+str(speed_value)#+str(flag_stop)
                direction = direction2


            elif (cmd_<=-10):
                command = "m1"  + "0" + str(abs(cmd_))+ "1"+str(speed_value)#+str(flag_stop)
                direction = direction0
            elif (cmd_>=10):
                command = "m1"  + "1" + str(abs(cmd_))+ "1"+str(speed_value)#+str(flag_stop)
                direction = direction1
            elif (cmd_>0):
                command = "m1" + "10" + str(abs(cmd_))+ "1"+str(speed_value)#+str(flag_stop)
                direction = direction1
            else:
                command = "m1" + "00" + str(abs(cmd_))+ "1"+str(speed_value)#+str(flag_stop)
                direction = direction0

                                                                                                                                                    
            cv2.putText(out_img1, 'turn :'+direction, (int(out_img1.shape[0]/2),50), cv2.FONT_HERSHEY_SIMPLEX , 1, (48, 255, 255), 2,cv2.LINE_AA)
            cv2.putText(out_img1, 'value:'+str(cmd_), (int(out_img1.shape[0]/2),80), cv2.FONT_HERSHEY_SIMPLEX , 1, (48, 255, 255), 2,cv2.LINE_AA)

            # if(warning_flag == 1):
            #     cv2.putText(out_img1, text_warning, (40,200), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 255), 2,cv2.LINE_AA)
            #显示速度
            cv2.putText(out_img1, 'Speed: '+str(speed_value), (20,110), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2,cv2.LINE_AA)
            

            
                


            cv2.imshow('raw img',img0)
            cv2.imshow('perspective1',out_img1)
            cv2.imshow('gray',gray)
            
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
                video_wirte.release()
                cap.release()
                cv2.destroyAllWindows()
                break
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
    exit()

    
        
            

                  