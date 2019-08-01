
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import threading
import _thread

import yaml

from my_threading import capturing_thread,processing_thread,Stack

if __name__ == '__main__':
    
    conf = yaml.load(open('config.yaml','r',encoding='utf-8'))
    path = conf['path']


    frame_buffer = Stack(4)
    threads = []
    t1 = threading.Thread(target = capturing_thread, args=(path, frame_buffer, threading.RLock()))
    t1.start()
    t2 = threading.Thread(target = processing_thread, args=(frame_buffer, threading.RLock()))
    t2.start()

    # threads.append(t1)
    # threads.append(t2)
    # # 等待所有线程完成
    # for t in threads:
    #     t.join()
    # print("主进程结束！")
