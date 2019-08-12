#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from motor import *
from socket import *
from threading import Thread
import threading
import sys, os 
from ev3dev2.sensor.lego import *

udp_socket = None
Addr = ('', 8888)
#RECV = True
lock = threading.Lock()

# 电机初始参数
new_steering_angle = 0
steering_angle = 0
speed_percent = 0

def udp_recv():
    print('udp_recv(%s)start' % (threading.current_thread().name))
    global new_steering_angle, steering_angle, speed_percent, RECV  
    while True:
        command = udp_socket.recv(1024).decode()
        print(command)
        if command[0:2] == 'm1':
            print("rev")
            if int(command[2:8]) == 0:
                print("stop")
                vehicle_stop()
                steer(0)
                new_steering_angle = 0
                steering_angle = 0
                speed_percent = 0
            else:
                print("rev speed")
                speed_percent = (int(command[5])*2-1) * int(command[6:8])
                new_steering_angle = (int(command[2])*2-1) * int(command[3:5])
                if new_steering_angle != steering_angle:
                    steering_angle = new_steering_angle
                    steer(steering_angle)
                print((speed_percent, steering_angle))
                go_straight(-speed_percent)
        else:
            pass

def program_restart():
    """软件重启"""
    python = sys.executable
    os.execl(python, python, * sys.argv)

def assistance():
    print('assistance thread(%s)start' % threading.current_thread().name)
    # PASS_WHITELINE = True
    global speed_percent
    # color_sensor = ColorSensor()
    touch_sensor = TouchSensor()
    # count_color = 0
    count_touch = 0
    while True:
        # if not PASS_WHITELINE:
        #     current_color = color_sensor.color
        #     print ("the current color is (%d)" % current_color)
        #     if (current_color == 2):  # 红蓝 2 红 6
        #         count_color += 1
        #     else:
        #         count_color = 0
        #     print ("the count of color is (%d)" % count_color)
        #     if (count_color == 2):
        #         lock.acquire()
        #         vehicle_stop()
        #         sleep(2)
        #         #go_straight(-speed_percent)
        #         PASS_WHITELINE = True
        #         lock.release()

        # touch sensor被按下，停车回正
        if (touch_sensor.is_pressed == 1): 
            print("stop")
            vehicle_stop()
            steer(0)
            new_steering_angle = 0
            steering_angle = 0
            speed_percent = 0
            count_touch += 1
        else:
            count_touch = 0
        if count_touch >= 25:
            """超过X秒，重启程序"""
            print('restarting......')
            program_restart()
        

if __name__ == '__main__':
    udp_socket = socket(AF_INET, SOCK_DGRAM)
    udp_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    udp_socket.bind(Addr)
    
    # 设置不同的线程
    thread_recv = Thread(target=udp_recv, name="udp_recv")
    thread_ass  = Thread(target=assistance, name="assistance")

    # 启动不同的线程
    thread_recv.start()
    thread_ass.start()

    # 阻塞主线程，直至子线程结束
    thread_recv.join()
    thread_ass.join()

    print('main thread(%s)end' %(threading.current_thread().name))







