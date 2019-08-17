# coding=utf-8
import numpy as np
import cv2
from time import sleep
import yaml

conf = yaml.load(open('config.yaml','r',encoding='utf-8'), Loader=yaml.FullLoader)
bais = conf['bais']
MIN_k = conf['MIN_k']
MAX_k = conf['MAX_k']

MAX_STEER0 = conf['MAX_STEER0']
MAX_STEER_0_1 = conf['MAX_STEER0_1']
MAX_STEER1 = conf['MAX_STEER1']

MAX_lane = conf['MAX_lane']


P_1 = conf['P_para1']

P_20 = conf['P_para20']
P_20_b = conf['P_para20_b']

P_21 =  conf['P_para21']
P_21_b = conf['P_para21_b']

lane_warning = conf['lane_warning']
P_all = conf['P_all']
delta_value = conf['delta_value']
_2_value_thread = conf['2_value_thread']
following_dotted_line = conf['dotted_line']


lower_white = np.array([0, 0, 220])  
upper_white = np.array([180, 30, 255]) 
h = 0
lower_yellow = np.array([26+h, 43, 46])  
upper_yellow = np.array([34+h, 255, 255]) 

#cv2.namedWindow("gray",cv2.WINDOW_NORMAL)





def img_hsv2gray_0(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR

    mask_white = cv2.inRange(hsv, lower_white, upper_white)  # 转换为hsv空间，去除背景
    # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_and(img,img, mask = mask_white)
    mask2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('mask2gray',mask2gray)
 
    #先膨胀后腐蚀
    kernel_dilated = cv2.getStructuringElement(cv2.MORPH_RECT, (10,15))# kernel = (20,20) 
    kernel_eroded = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilated = cv2.dilate(mask2gray, kernel_dilated)
    eroded = cv2.erode(dilated, kernel_eroded)
    #eroded = cv2.morphologyEx(mask2gray, cv2.MORPH_OPEN, kernel)  #先膨胀后腐蚀  效果不好
    # cv2.imshow('eroded',eroded)
    
    return eroded

# yellow is open
def img_hsv2gray_1(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hsv色彩空间 CV_HSV2BGR

    mask_white = cv2.inRange(hsv, lower_white, upper_white)  # 转换为hsv空间，去除背景
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_and(img,img, mask = mask_white + mask_yellow)
    mask2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('mask2gray',mask2gray)
 
    #先膨胀后腐蚀
    kernel_dilated = cv2.getStructuringElement(cv2.MORPH_RECT, (10,15))# kernel = (20,20) 
    kernel_eroded = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilated = cv2.dilate(mask2gray, kernel_dilated)
    eroded = cv2.erode(dilated, kernel_eroded)
    #eroded = cv2.morphologyEx(mask2gray, cv2.MORPH_OPEN, kernel)  #先膨胀后腐蚀  效果不好
    # cv2.imshow('eroded',eroded)

    return eroded
   

def find_line(gray):

    margin = 100  #滑窗x范围
    minpix = 50   

    # Take a histogram of the bottom half of the image
    histogram = np.sum(gray[gray.shape[0]//2:,:], axis=0)
  
    midpoint = np.int(histogram.shape[0]/2) + bais
    leftx_base = np.argmax(histogram[:midpoint])#左侧找最大峰值点位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint#右侧找最大峰值点位置
    # cv2.circle(out_img1,(leftx_base,380), 10, (0,0,255), -1)
    # cv2.circle(out_img1,(rightx_base,380), 10, (0,0,255), -1)
    # print('midpoint:',midpoint)
    
    global window_height,nwindows
    nwindows = 9
    window_height = np.int(gray.shape[0]/nwindows)
  
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = gray.nonzero()
    global nonzeroy,nonzerox
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    #print('leftx_base:',leftx_base,'rightx_base:',rightx_base,'midpoint:',midpoint)
   
    left_lane_inds = []
    right_lane_inds = []
    
    left_center = []
    right_center = []

    left = []
    right = []

    flag = 0
    flag0 = 0
    flag1 = 0
       
    # Step through the windows one by one
    for k in range(nwindows):

        win_y_low = gray.shape[0] - (k+1)*window_height
        win_y_high = gray.shape[0] - k*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # cv2.rectangle(out_img1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img1, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
        # cv2.imshow('hhh',out_img1)
        #满足条件的索引     #非0像素点的y坐标
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #y_current = int((win_y_low+win_y_high)/2)

        # 计算得到左右框的中心 x坐标
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))#求平均
            lefty_current = np.int(np.mean(nonzeroy[good_left_inds]))
            left_center.append(np.array([leftx_current,lefty_current]))
        else:
            lefty_current = int((win_y_low+win_y_high)/2)


        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            righty_current = np.int(np.mean(nonzeroy[good_right_inds]))
            right_center.append(np.array([rightx_current,righty_current]))
        else:
            righty_current = int((win_y_low+win_y_high)/2)
        
 
        # if(gray[y_current,leftx_current] > 0):
        #     left_center.append(np.array([leftx_current,y_current]))
        #     left.append(np.array([leftx_current,y_current,1]))
        # else:
        #     left.append(np.array([leftx_current,y_current,0]))
        #     #print('gray:',gray[y_current,leftx_current])

        # if(gray[y_current,rightx_current] > 0 ):
        #     right_center.append(np.array([rightx_current,y_current]))
        #     right.append(np.array([rightx_current,y_current,1]))
        # else:
        #     right.append(np.array([rightx_current,y_current,0]))
        left.append(np.array([leftx_current,lefty_current,1]))
        right.append(np.array([rightx_current,righty_current,0]))
 

    # for i in range(0,len(left_center)):
    #     print('left_center:',left_center[i][0],left_center[i][1])
    # for i in range(0,len(right_center)):
    #     print('right_center:',right_center[i][0],right_center[i][1])
    # print('----------------------------------------------------------')
    if(len(left_center) <= 2): 
        left_center = []
        left = []
    if(len(right_center) <= 2): 
        right_center = []
        right = [] 
    

    if(len(left_center) > 0):   
        if((left[0][0] > midpoint)| (left_center[0][0] > midpoint) | (leftx_base == midpoint)):
            left_center = []
            left = []
    if(len(right_center) > 0):
        if((right[0][0] < midpoint)| (right_center[0][0] < midpoint)| (rightx_base == midpoint)):
            right_center = []
            right = []
 

    nums_ = 0
    if((len(left_center)==9) & (len(right_center)==9)):
        for i in range(0,9):
            if((left_center[i][0]-right_center[i][0])==0):
                nums_ = nums_+1
        if(nums_ >= 2):
            if(left_center[0][0] < midpoint):#&(left_center[1][0] < midpoint)):
                right_center = []
                right = []
            print('right_center : ---------',len(right_center))
            if((len(right_center) > 0)):
                if((right_center[0][0] > midpoint)):#&(right_center[1][0] > midpoint)):
                    left_center = []
                    left = []


   
    if(len(left_center)>0):
        nums_ = 0
        for i in range(1,len(left_center)):
            if((left_center[i-1][1] - left_center[i][1]) > (3.5*window_height)):
                left_center = []
                left =[]
                break
            if((left_center[i-1][1] - left_center[i][1]) > (1.5*window_height)):
                nums_ = nums_+1;                
            
            if(nums_ >=1):
                flag0 = 2
            if(nums_ == 0 and len(left_center) == nwindows):
                flag0 = 1

    
    if(len(right_center)>0): 
        nums_ = 0      
        for i in range(1,len(right_center)):
            if((right_center[i-1][1] - right_center[i][1]) > (3.5*window_height)):
                right_center = []
                right =[]
                break
            if((right_center[i-1][1] - right_center[i][1]) > (1.5*window_height)):
                nums_ = nums_+1
            
            if(nums_ >=1):
                flag1 = 1
            if(nums_ == 0 and len(right_center)== nwindows):
                flag1 = 2


    if(((flag0 != 0)&(flag1 == 0)) |((flag0 != 0)&(flag0 == flag1))):
        flag = flag0
    elif((flag0 == 0)&(flag1 != 0)):
        flag = flag1
    else:
        flag = 0

    # for i in range(0,len(left_center)):
    #     print('left_center:',left_center[i][0],left_center[i][1])
    # for i in range(0,len(right_center)):
    #     print('right_center:',right_center[i][0],right_center[i][1])
    # print('leftx_base:',leftx_base)
    # print('rightx_base:',rightx_base)
    # print('midpoint:',midpoint)
    # print('window_height:',window_height)
            
          
    return left_center, right_center,flag,left,right



 

# python不允许选择采用传值还是传引用。
# Python参数传递采用的肯定是“传对象引用”的方式。这种方式相当于传值和传引用的一种综合。
def lane_keeping(img_roi,flag_t,steering_memory,lane_memory,change_cmd,yellow_is_open):
    global STEER_limited,k_value_inv,d_from_center,out_img1

    if(yellow_is_open == 1):
        gray = img_hsv2gray_1(img_roi)
    else:
        gray = img_hsv2gray_0(img_roi)
    #二值化处理
    #_,gray = cv2.threshold(gray, _2_value_thread, 255, cv2.THRESH_BINARY) 

    #cv2.imshow('gray1_2__',gray)

    #kernel_size = 5
    #gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1.5)
    #low_threshold = 50
    #high_threshold = 150
    #edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    xmax = img_roi.shape[1]
    xmin = 0

    ymax = img_roi.shape[0]
    ymin = 0

    w = 510
    h = 500


    if(flag_t == 0):
        # a = int(w/3)
        # b = 90
        # c = h+80

        # d = int(w/3*2)

        # k1 = 10
        # k2 = 30
        # point1 = np.array([[130+k1,70],[60+k2,300],[480-k1,70],[550-k2,300]],dtype = "float32")
        # point2 = np.array([[a,b],[a,c],[d,b],[d,c]],dtype = "float32")
        # M = cv2.getPerspectiveTransform(point1,point2)
        # Minv = cv2.getPerspectiveTransform(point2,point1)

        pts1 = np.float32([[163, 1], [416, 1], [49, 479], [518, 479]])
        ab = 130            
        ac = 0            
        al = 500            
        ak = 200            
        pts2 = np.float32([[ab, ac], [ab + ak, ac], [ab, al + ac], [ab + ak, al + ac]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        Minv = cv2.getPerspectiveTransform(pts2, pts1)
        flag_t = 1
    


    # out_img = cv2.warpPerspective(gray,M,(w,h))
    # out_img1 = cv2.warpPerspective(img_roi,M,(w,h))
    out_img = cv2.warpPerspective(gray,M,(w,h))
    out_img1 = cv2.warpPerspective(img_roi,M,(w,h))

    #out_img1 = cv2.warpPerspective(out_img,Minv,(w,h))
    #cv2.imshow('gray',out_img)#看透视变换后灰度图-------------good

    left_center,right_center,lane_flag,left_center0,right_center0 = find_line(out_img)
    len_l = len(left_center)
    len_r = len(right_center)
    print('len_l:',len_l,'len_r:',len_r)
    print('len_l:',len(left_center0),'len_r:',len(right_center0))

    if((lane_flag == 1) | (lane_flag == 2)):
        current_lane = lane_flag
    else:
        current_lane = lane_memory

    if(current_lane == 2):#位于右道
        #print('left is xu , right is shi')
        l_color = (255, 0, 0)
        r_color = (0, 0, 255)
        cv2.putText(out_img1, 'R', (20,50), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 255), 2,cv2.LINE_AA)
        
    else:
        cv2.putText(out_img1, 'L', (20,50), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 0, 255), 2,cv2.LINE_AA)
        l_color = (0, 0, 255)
        r_color = (255, 0, 0)
    

    for i in range(0,len(right_center)):
        # print('right_center:',right_center[i][0],right_center[i][1])#,right_center[i][2])
        #if(right_center[i][2] == 1):
        cv2.circle(out_img1, (right_center[i][0],right_center[i][1]), 5, r_color, -1)
    
    for i in range(0,len(left_center)):
        # print('left_center:',left_center[i][0],left_center[i][1])#,left_center[i][2])
        #if(left_center[i][2] == 1):
        cv2.circle(out_img1, (left_center[i][0],left_center[i][1]), 5, l_color, -1)
       
    
            
    
            

    #print('window_height:',window_height)

    x_vision = int(out_img1.shape[1]/2) + bais #视野中心线
    print('x_vision:',x_vision)
    cv2.line(out_img1, (x_vision,360), (x_vision,h), (255, 255, 0), 5)

    MAX_STEER = 0
    k_value_inv = 0
    d_from_center = 0
    cmd_ = 0
    lane_inv_2 = 140
    target_line = np.zeros((nwindows,2),dtype = "int")
    dotted_line = np.zeros((nwindows,2),dtype = "int")

    index_ = 2
#-------------------计算斜率和偏差---------------------------
    if((len_l>0) & (len_r>0)):

        for i in range(0,nwindows):
            target_line[i][0] = int((left_center0[i][0]+right_center0[i][0])/2)
            target_line[i][1] = int((left_center0[i][1]+right_center0[i][1])/2)
            
            
        p1 = (target_line[index_][0],target_line[index_][1]) 
        p2 = (x_vision,h) 
        #cv2.line(out_img1, p1, p2, (0, 255, 0), 5)

        # index = min(len_l,len_r)-1

        #-----------------变道指令--------------
        

        if((change_cmd == 1) & (current_lane != 1)):
            for i in range(len(target_line)):
                target_line[i][0] = target_line[i][0] - 2 * lane_inv_2
        if((change_cmd == 2) & (current_lane != 2)):
            for i in range(len(target_line)):
                target_line[i][0] = target_line[i][0] + 2 * lane_inv_2
          
        #--------------------------------------



        top_target = target_line[index_][0]
        bottom_target = target_line[0][0]
        
        d_from_center = top_target - x_vision
        if(change_cmd == 0):
            STEER_limited = 14
        else:
            STEER_limited = MAX_STEER0

        cmd_ = d_from_center * P_1 

        print('-- calculated by d_from_center --' )

        cv2.line(out_img1, (target_line[index_][0], target_line[index_][1]), (x_vision,h), (0, 255, 0), 5)


    elif((len_l > 0) | (len_r > 0)):  #只识别到1根线
    #规划轨迹
        # p_top = []
        # p_bottom = []
        if((len_l > 0) & (current_lane == 1)):
            # right_center = np.zeros((nwindows,2),dtype = "int")
            for i in range(0,nwindows):
                dotted_line[i][0] = left_center0[i][0] + 2 * lane_inv_2
                dotted_line[i][1] =  left_center0[i][1]
                cv2.circle(out_img1, (dotted_line[i][0],dotted_line[i][1]), 5, r_color, -1)
            right_center = dotted_line
     
            for i in range(0,nwindows):
                target_line[i][0] = int((left_center0[i][0]+dotted_line[i][0])/2)
                target_line[i][1] = int((left_center0[i][1]+dotted_line[i][1])/2)
         
            #top_target = [target_line[len(target_line)-1][0] , target_line[len(target_line)-1][1]]


        elif((len_l > 0) & (current_lane == 2)):
            # p_top = [left_center[len_l-1][0],left_center[len_l-1][1]]
            # p_bottom = [left_center[0][0],left_center[0][1]]
            dotted_line = left_center0
            right_center = np.zeros((nwindows,2),dtype = "int")
            for i in range(0,nwindows):
                right_center[i][0] = left_center0[i][0] + 2 * lane_inv_2
                right_center[i][1] =  left_center0[i][1]
                cv2.circle(out_img1, (right_center[i][0],right_center[i][1]), 5, r_color, -1)

            
            for i in range(0,nwindows):
                target_line[i][0] = int((left_center0[i][0]+right_center[i][0])/2)
                target_line[i][1] = int((left_center0[i][1]+right_center[i][1])/2)

            #top_target = [target_line[len(target_line)-1][0], target_line[len(target_line)-1][1]]

        elif((len_r > 0) & (current_lane == 2)):
            # p_top = [right_center[len_l-1][0],right_center[len_l-1][1]]
            # p_bottom = [right_center[0][0],right_center[0][1]]
            #left_center = np.zeros((nwindows,2),dtype = "int")
         
            for i in range(0,nwindows):
                dotted_line[i][0] = right_center0[i][0] - 2 * lane_inv_2
                dotted_line[i][1] =  right_center0[i][1]
                cv2.circle(out_img1, (dotted_line[i][0],dotted_line[i][1]), 5, l_color, -1)
            left_center = dotted_line
        
            for i in range(0,nwindows):
                target_line[i][0] = int((right_center0[i][0]+dotted_line[i][0])/2)
                target_line[i][1] = int((right_center0[i][1]+dotted_line[i][1])/2)
            #top_target = [target_line[len(target_line)-1][0] , target_line[len(target_line)-1][1]]

       
        elif((len_r > 0) & (current_lane == 1)):
            # p_top = [right_center[len_l-1][0],right_center[len_l-1][1]]
            # p_bottom = [right_center[0][0],right_center[0][1]]
            left_center = np.zeros((nwindows,2),dtype = "int")
            dotted_line = right_center0
            for i in range(0,nwindows):
                left_center[i][0] = right_center0[i][0] - 2 * lane_inv_2
                left_center[i][1] =  right_center0[i][1]
                cv2.circle(out_img1, (left_center[i][0],left_center[i][1]), 5, l_color, -1)

          
            for i in range(0,nwindows):
                target_line[i][0] = int((right_center0[i][0]+left_center[i][0])/2)
                target_line[i][1] = int((right_center0[i][1]+left_center[i][1])/2)
            #top_target = [target_line[len(target_line)-1][0] , target_line[len(target_line)-1][1]]

       
        else:
            ttt = 1


        #-----------------变道指令--------------
        if((change_cmd == 1) & (current_lane != 1)):
            for i in range(len(target_line)):
                target_line[i][0] = target_line[i][0] - 2 * lane_inv_2
        if((change_cmd == 2) & (current_lane != 2)):
            for i in range(len(target_line)):
                target_line[i][0] = target_line[i][0] + 2 * lane_inv_2
        
        #--------------------------------------


       
        top_target = [target_line[index_][0] , target_line[index_][1]]
        bottom_target = [x_vision,h]

        #cv2.line(out_img1, (target_line[nwindows-1][0], target_line[nwindows-1][1]), (x_vision,h), (0, 255, 0), 5)
        cv2.line(out_img1, (target_line[index_][0], target_line[index_][1]), (x_vision,h), (0, 255, 0), 5)
               

        #d_y = p_top[1]-p_bottom[1]
        d_y = top_target[1]-bottom_target[1]
        if(d_y == 0):
            d_y = 0.0000001
        k_value_inv = round(-(top_target[0]-bottom_target[0])/(d_y),4)
        #k_value_inv = round(-(p_top[0]-p_top[0])/(d_y),4)
   
        if(abs(k_value_inv) > MAX_k):                # (k > MAX_k)
            STEER_limited = MAX_STEER1                 # 过急弯时把约束最大转角放大
            cmd_ = k_value_inv*P_21 
            if(k_value_inv < 0):
                cmd_ = cmd_ - P_21_b
            else:
                cmd_ = cmd_ + P_21_b
            print('-- calculated by k_value_inv --' )
            print('--- part 3 : (abs(k) > ', MAX_k, ' --\n' )
      
        else:                                          # (0 < k < MAX_k)
            STEER_limited = MAX_STEER_0_1 
            cmd_ = (P_20 * k_value_inv) * (P_20 * k_value_inv)  + P_20_b   #采用平方计算策略
            #cmd_ = P_20 * k_value_inv                                      #采用比例计算策略
            if(k_value_inv < 0):
                cmd_ = - cmd_
            else:
                cmd_ = cmd_
            print('-- calculated by k_value_inv --' )
            print('-- part 2 :',MIN_k,'< (abs(k) <', MAX_k, ' --\n' )
 
    
    else:
        STEER_limited = 0
        d_from_center = 0
        k_value_inv = 0
        cmd_ = steering_memory
        print('--------------- Null in k_value_inv & d_from_center ---------------')
    
    #cv2.putText(out_img1, 'target_line: '+ str(len(target_line)) , (10,280), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2,cv2.LINE_AA)
    for i in range(0,index_):
        print('target_line:',target_line[i])

    for i in range(0,len(target_line)):
        cv2.circle(out_img1, (target_line[i][0],target_line[i][1]), 5, (0, 255, 0), -1)

        #-----------第1层预警-----------
    d_warning_top = abs(target_line[index_][0] - x_vision)
    d_warning_bottom =  d_warning_bottom1 = abs(target_line[0][0] - x_vision)

    print('d_warning_top:',d_warning_top)
    print('d_warning_bottom:',d_warning_bottom)


    if(((d_warning_top > (0.9*lane_warning)) | (d_warning_bottom > (0.9*lane_warning))) & (k_value_inv != 0)):
        if((d_warning_top > lane_warning) | (d_warning_bottom > lane_warning)):
            cmd_ = MAX_STEER1 * k_value_inv / abs(k_value_inv)
            STEER_limited = MAX_STEER1
            cv2.putText(out_img1, 'Turn '+ str(MAX_STEER1) +' !!', (20,80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2,cv2.LINE_AA)
        else:
            cmd_ = MAX_STEER_0_1 * k_value_inv / abs(k_value_inv)
            STEER_limited = MAX_STEER_0_1
            cv2.putText(out_img1, 'Turn '+str(MAX_STEER_0_1) +' !!', (20,80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2,cv2.LINE_AA)

    
    
    print('Current STEER_limited = ',STEER_limited)

    print("cmd_          = ",round(cmd_,4))
    print('d_from_center = ',d_from_center)
    print('k_value_inv   = ',k_value_inv)

       
    cmd_ = int( cmd_)

    if(abs(cmd_)>STEER_limited):
        if(cmd_>0):
            cmd_ = STEER_limited
        if(cmd_<0):
            cmd_ = -STEER_limited
    
    return cmd_ ,current_lane, out_img1,out_img
