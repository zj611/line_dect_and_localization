# coding=utf-8
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import threading
#import _thread

import datetime
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import yaml

from function.my_threading import Stack,capture_thread2,capture_thread0,process_thread0


class mywindow(QtWidgets.QWidget):

    def __init__(self):
        QDialog.__init__(self)
        self.initUI()

    def initUI(self):
        self.controlsGroup = QGroupBox()
        self.lcdNumber = QLCDNumber(self)
        self.lcdNumber.setFixedSize(150,50)

        vbox = QVBoxLayout()

        vbox.addWidget(self.lcdNumber)
 
        self.controlsGroup.setLayout(vbox)

        controlsLayout = QGridLayout()
        self.line_steer = QLineEdit() 
        self.line_steer.setFixedSize(130,30)
        # l1.move(100,20)

        self.line_speed = QLineEdit() 
        self.line_speed.setFixedSize(130,30)
        self.steerLabel = QLabel("转向指令:")
        self.speedLabel = QLabel("速度指令:")
        self.logoLabel = QLabel()
        self.imgLabel = QLabel()
        # self.logoLabel.setFixedSize(10,10)
        self.logoLabel.move(1,1)
        

        self.buttonStart = QPushButton("start")
        self.buttonStop = QPushButton("stop")
  

        # fname, _ = QFileDialog.getOpenFileName(self, '选择图片', './', 'timg.jpeg')
        # self.label2.setPixmap(QPixmap(fname))
        
        controlsLayout.addWidget(self.line_steer,1,1)
        controlsLayout.addWidget(self.line_speed,0,1)
        controlsLayout.addWidget(self.steerLabel,1,0)
        controlsLayout.addWidget(self.speedLabel,0,0)  
        
        controlsLayout.addWidget(self.imgLabel,2,1)
        controlsLayout.addWidget(self.buttonStop,1,2)
        controlsLayout.addWidget(self.buttonStart,2,2)

        # png = QtGui.QPixmap('./timg.jpeg')

        # # 在l1里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
        # self.logoLabel.setPixmap(png)

        img = QImage('./function/timg.jpeg')  #创建图片实例
        mgnWidth = int(img.height() * 0.6)    
        mgnHeight = int(img.width() * 0.6)    #缩放宽高尺寸
        size = QSize(mgnWidth, mgnHeight)
    
        pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio))       #修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中
        vbox.addWidget(self.logoLabel)
        self.logoLabel.resize(mgnWidth, mgnHeight)
        self.logoLabel.setPixmap(pixImg)
        # self.logoLabel.move(700,600)

        layout = QHBoxLayout()
        layout.addWidget(self.controlsGroup)
        layout.addLayout(controlsLayout)
        self.setLayout(layout)


        # self.slider.valueChanged.connect(self.lcdNumber.display)
        # self.buttonSave.clicked.connect(self.showMessage)
        # self.buttonRun.clicked.connect(self.showMessage)
   
        self.buttonStart.clicked.connect(self.start_)
        self.buttonStop.clicked.connect(self.stop_)
        self.setGeometry(300, 500, 500, 180)
        self.setWindowTitle('SAIC上汽软件大赛 乘游记队 ')

        

 
 
    def start_(self):
        frame_buffer = Stack(8)
        conf = yaml.load(open('config.yaml','r',encoding='utf-8'), Loader=yaml.FullLoader)
        path = conf['path']
        qmut_1 = QMutex()
        qmut_2 = QMutex()
        self.cap_thread = capture_thread0(path,frame_buffer,qmut_1)#threading.RLock()
        #self.cap_thread = capture_thread2(path,frame_buffer, qmut_1)
        self.pro_thread = process_thread0(frame_buffer, qmut_2)

        self.pro_thread._signal.connect(self.callbacklog)
        self.pro_thread._signal_img.connect(self.callbackImg)
        self.cap_thread.start()
        self.pro_thread.start()

    def stop_(self):
        self.cap_thread.exit()
        self.pro_thread.exit()
        exit(-1)


 
    def callbacklog(self, steer,speed):
        self.line_steer.setText(steer)
        self.line_speed.setText(speed)
        self.lcdNumber.display(steer)

    def callbackImg(self,img): 
        mgnWidth = int(img.height() * 0.5)    
        mgnHeight = int(img.width() * 0.5)    #缩放宽高尺寸
        size = QSize(mgnWidth, mgnHeight)    
        pixImg = QPixmap.fromImage(img.scaled(size, Qt.IgnoreAspectRatio)) #修改图片实例大小并从QImage实例中生成QPixmap实例以备放入QLabel控件中
        self.imgLabel.resize(mgnWidth, mgnHeight)
        self.imgLabel.setPixmap(pixImg)
          
       

 
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    m = mywindow()
    m.show()
    sys.exit(app.exec_())

