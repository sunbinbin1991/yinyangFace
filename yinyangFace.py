#coding=utf-8
import cv2
import mxnet as mx
import numpy as np
from multiprocessing import Process, freeze_support
from collections import namedtuple
from mtcnn_detector import MtcnnDetector
import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX
import glob as gb
color = (255, 64, 128)
def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    point = [x,y]
    return point

def location_p(p1,p2,p3):
    '''
    p3 zai (p1,p2) shang de ying she dian
    :param p1:
    :param p2:
    :param p3:
    :return:
    '''
    p=[]
    x = 0
    y = 0
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    x3 = p3[0]
    y3 = p3[1]
    if y1 == y2:
        y = y1
        x = x3
    else:
        k=(x1-x2)*1.0/(y1-y2)
        y = (k*k*y1-k*x1+k*x3+y3)*1.0/(k*k+1)
        x = x1 + k*(y-y1)

    p = np.array([x,y])
    return p

def estimate_5points(shape):
    '''
    :param Img:
    :param points_5: 5*2  left_eye right_eye noise left_mouse right_mouse
    :return:
    '''
    left_middle = np.array(shape[36:42]).mean(axis=0)
    right_middle = np.array( shape[42:48]).mean(axis=0)
    nose_part = np.array(shape[30:36]).mean(axis=0)
    left_mouth = np.array(shape[48])
    right_mouth = np.array(shape[54])
    s = np.append(left_middle,right_middle,axis=0)
    s = np.append(s,nose_part,axis=0)
    s = np.append(s,left_mouth,axis=0)
    s = np.append(s,right_mouth,axis=0)
    return s

def demo_keypointfive():
    img_path = gb.glob("../image/*.jpg")
    for path in img_path:
        img = cv2.imread(path)
        cv2.imshow("img",img)
        h,w, = img.shape[:2]
        Ey= 0.4
        Eeye_x = 1.6
        Emouth_x = 1.4
        mask =  np.zeros([h, w], dtype = np.uint8)
        mask2 =  np.zeros([h, w], dtype = np.uint8)
        area =  np.ones([h, w], dtype = np.uint8)
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        det_result = detector.detect_face(img)
        if det_result is not None:
            boxes, points = det_result
            for point in points:
                pset_x = point[0:5]
                pset_y = point[5:10]
                left_eye = [pset_x[0],pset_y[0]]
                right_eye = [pset_x[1],pset_y[1]]
                nose = [pset_x[2],pset_y[2]]
                left_mouth = [pset_x[3],pset_y[3]]
                right_mouth = [pset_x[4],pset_y[4]]
                center_eye = [left_eye[0]+(right_eye[0]-left_eye[0])/2,left_eye[1]+(right_eye[1]-left_eye[1])/2]
                center_mouth = [left_mouth[0]+(right_mouth[0]-left_mouth[0])/2,left_mouth[1]+(right_mouth[1]-left_mouth[1])/2]
                extend_y =Ey*(center_mouth[1]-center_eye[1])
                extend_eyex =Eeye_x*(center_eye[0]-left_eye[0])
                extend_mouthx =Emouth_x*(center_mouth[0]-left_mouth[0])

                point1 = np.array([1,0])
                point2 = np.array([center_mouth[0]-center_eye[0],center_mouth[1]-center_eye[1]])
                Lx = np.sqrt(point1.dot(point1))
                Ly = np.sqrt(point2.dot(point2))
                cos_a = point2.dot(point1) / (Lx * Ly)
                theta = np.arccos(cos_a)
                slant_x_extend = extend_y*cos_a
                slant_y_extend = extend_y*np.sin(theta)
                new_center_eye= [center_eye[0]-slant_x_extend, center_eye[1] - slant_y_extend]
                eye_x_extend= extend_eyex*np.sin(theta)
                eye_y_extend= extend_eyex*cos_a
                new_left_eye = [new_center_eye[0] - eye_x_extend, new_center_eye[1] + eye_y_extend]
                # print new_center_eye,new_left_eye2
                mouth_x_extend= extend_mouthx*np.sin(theta)
                mouth_y_extend= extend_mouthx*cos_a
                new_center_mouth = [center_mouth[0] +slant_x_extend, center_mouth[1] + slant_y_extend]
                new_left_mouth = [new_center_mouth[0] - mouth_x_extend, new_center_mouth[1] + mouth_y_extend]
                new_right_eye = [new_center_eye[0] + eye_x_extend, new_center_eye[1] - eye_y_extend]
                new_right_mouth = [new_center_mouth[0] + mouth_x_extend, new_center_mouth[1] -mouth_y_extend]
                # print new_center_mouth, new_left_mouth
                cv2.circle(img, ((int)(new_center_eye[0]), (int)(new_center_eye[1])), 2, (255, 0, 0))
                cv2.circle(img, ((int)(new_left_eye[0]), (int)(new_left_eye[1])), 2, (255, 0, 0))
                cv2.circle(img, ((int)(new_center_mouth[0]), (int)(new_center_mouth[1])), 2, (255, 0, 0))
                cv2.circle(img, ((int)(new_left_mouth[0]), (int)(new_left_mouth[1])), 2, (255, 0, 0))
                cv2.circle(img, ((int)(new_right_eye[0]), (int)(new_right_eye[1])), 2, (255, 0, 255))
                cv2.circle(img, ((int)(new_right_mouth[0]), (int)(new_right_mouth[1])), 2, (255, 0, 255))

                leftface = np.array([[[new_left_eye[0], new_left_eye[1]], [new_center_eye[0], new_center_eye[1]],  [new_center_mouth[0], new_center_mouth[1]],[new_left_mouth[0], new_left_mouth[1]]]], dtype=np.int32)
                rightface = np.array([[[new_right_eye[0], new_right_eye[1]], [new_center_eye[0], new_center_eye[1]],  [new_center_mouth[0], new_center_mouth[1]],[new_right_mouth[0], new_right_mouth[1]]]], dtype=np.int32)

                cv2.fillPoly(mask, leftface, 255)
                cv2.fillPoly(mask2, rightface, 255)
                leftarea = np.sum(mask)/255.
                rightarea = np.sum(mask2)/255.
                print leftarea,rightarea
                leftmask = cv2.bitwise_and(img, img, mask=mask)
                # roi = cv2.multiply(img, mask)
                rightmask = cv2.bitwise_or(img, img, mask=mask2)

                leftmask_hsv = cv2.cvtColor(leftmask, cv2.COLOR_BGR2HSV)
                rightmask_hsv = cv2.cvtColor(rightmask, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(leftmask_hsv)
                h2, s2, v2 = cv2.split(rightmask_hsv)
                # rate = np.sum(v)*1.0/np.sum(v2)
                leftvrate= np.sum(v) * 1.0 /leftarea
                rightRate = np.sum(v2)*1.0/rightarea
                rate = abs(leftvrate / (leftvrate + rightRate) - 0.5)
                print rate
                cv2.imshow("mask", mask)
                cv2.imshow("v", v)
                cv2.imshow("v2", v2)
                cv2.imshow("img", img)
                cv2.waitKey(0)

def demo_keypoint68():
    from detect.dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    detector = cv_get_frontal_face_detector()
    # img = cv2.imread("../image/3.jpg")
    # img_path = gb.glob(r"Y:\maxiaofang\data\Uface_register_face\*g")
    img_path = gb.glob(r"Y:\maxiaofang\data\10000\tight\suc\*g")
    img_path = gb.glob(r"./reg/0.2-0.8/light/*g")

    # img_path = gb.glob(r"./image/*g")
    # img_path = gb.glob(r"D:\face_lib\CelebA\Img\img_align_celeba_png.7z\img_align_celeba_png/*g")
    histLength = len(img_path)
    yinyangScore = np.zeros(histLength)
    personNum = 0
    lightcount =1
    for path in img_path:
        # print path
        img = cv2.imread(path)
        # img = cv2.imread(r"Y:\maxiaofang\data\all\light\05B1C54C697240228A6C62111FF6AB0D.jpg")
        # img = cv2.imread(r"./image/0.93_267.jpg")
        h,w = img.shape[:2]
        mask =  np.zeros([h, w], dtype = np.uint8)
        mask2 =  np.zeros([h, w], dtype = np.uint8)
        maskw =  np.zeros([h, w], dtype = np.uint8)
        # for some reason some too big picture may could not find face,so the shape will devided
        shapes, rects = predictor(img)
        for shape in shapes:
            keypoint_68= []
            x1,y1 = np.min(shape, axis=0)
            x2,y2 = np.max(shape, axis=0)
            mean_pixels = (127.5)
            # # _mean_pixels = np.array(mean_pixels).reshape((1, 2))
            # img_var = img[y1:y2,x1:x2].copy()
            # img_var = cv2.cvtColor(img_var, cv2.COLOR_BGR2HSV)
            # img_varH, img_varS, img_varV = cv2.split(img_var)
            # data = img_varV.astype('float32')
            # _data = (data - np.array(mean_pixels)) * 1 / 127.5
            # cv2.imshow("var",img[y1:y2,x1:x2])
            # print "np.var = ",np.var(_data)
            keypointFive = estimate_5points(shape)
            if True:
                whole =[]
                for pt in shape[:17]:
                    whole.append(pt)
                for i in range(22, 27):
                    whole.append(shape[22 + 26 - i])
                for i in range(17, 21):
                    whole.append(shape[17 + 21 - i])
                wholeface = np.array([[whole]], dtype=np.int32)
                cv2.fillPoly(maskw, wholeface, 255)
                wholemask = cv2.bitwise_and(img, img, mask=maskw)
                wholemask_hsv = cv2.cvtColor(wholemask, cv2.COLOR_BGR2HSV)
                h,s,v = cv2.split(wholemask_hsv)
                facearea = np.sum(maskw)/255.
                mean_v = np.sum(v) * 1.0 / facearea/255.
                mean_b = np.sum(wholemask[:,:,0]) * 1.0 / facearea
                mean_g = np.sum(wholemask[:,:,1]) * 1.0 / facearea
                mean_r = np.sum(wholemask[:,:,2]) * 1.0 / facearea
                print mean_b,mean_g,mean_r
                # vs= v.astype("float32")
                # v_tmp = vs*0.2
                # v_tmp2 = v_tmp.astype("uint8")
                print "wholemask_v= ",mean_v,mean_v*255
                # img2 = cv2.merge([h,s,v_tmp2])
                # if mean_v<0.88 and mean_v>0.2:
                # if  mean_v>0.4:
                #     continue
                # cv2.imwrite("./reg/%.2f_%d.jpg"%(mean_v,lightcount),img)
                lightcount+=1
                # draw point cloud
                # cv2.circle(wholemask, (50,50), 20, (170,220,255), thickness=-1)
                cv2.imshow("wholemask",wholemask)
                cv2.imshow("wholemask_hsv",wholemask_hsv)
                # cv2.imshow("v",v)
                # cv2.imshow("v_tmp",v_tmp2)
                # cv2.imshow("img2",img2)
                # for partshape in whole:
                    # cv2.circle(img, tuple(partshape), 2, color, thickness=2)
            for i in xrange(5):
                x1= keypointFive[2*i]
                y1= keypointFive[2*i+1]
                # cv2.circle(img, ((int)(x1),(int)(y1)), 2, color, thickness=2)
                # cv2.imshow("img", img)
                # # cv2.imshow("mask", leftmask)
                # # cv2.imshow("mask2", rightmask)
                # cv2.waitKey(0)
            left = []
            for pt in shape[:9]:
                left.append(pt)
            left.append(shape[28])
            left.append(shape[27])
            for i in range(17,21):
               left.append(shape[17+21-i])
            leftface = np.array([[left]],dtype=np.int32)
            cv2.fillPoly(mask, leftface, 255)
            leftmask = cv2.bitwise_and(img, img, mask=mask)
            right =[]
            for pt in shape[8:16]:
                right.append(pt)
            right.append(shape[26])
            right.append(shape[24])
            right.append(shape[22])
            for pt in shape[27:30]:
                right.append(pt)
            # right.append(shape[9])
            right = np.array([[right]],dtype=np.int32)
            cv2.fillPoly(mask2, right, 255)
            rightmask = cv2.bitwise_and(img, img, mask=mask2)
            leftmask_hsv = cv2.cvtColor(leftmask, cv2.COLOR_BGR2HSV)
            rightmask_hsv = cv2.cvtColor(rightmask, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(leftmask_hsv)
            h2, s2, v2 = cv2.split(rightmask_hsv)
            leftarea = np.sum(mask) / 255.
            # print  " np.sum(mask) ",np.sum(v),"_leftarea:",leftarea
            rightarea = np.sum(mask2) / 255.
            leftvrate = np.sum(v) * 1.0 / leftarea
            rightRate = np.sum(v2) * 1.0 / rightarea
            # print  "avr =",(np.sum(v) * 1.0 +np.sum(v2) * 1.0)/ (leftarea+  rightarea)
            # print leftarea, rightarea,"leftRate==",leftvrate,"rightRate==",rightRate
            # print abs(leftvrate-rightRate)
            rate = abs(leftvrate / (leftvrate + rightRate) - 0.5)
            # if rate<0.1:
            #     continue
            # rate = abs(np.sum(v) * 1.0 / (np.sum(v2)+ np.sum(v))-0.5)
            # print rate

            # draw point cloud
            for shape in shapes:
                count = 0
                for pt in shape:
                    # cv2.circle(img, tuple(pt), 2, color, thickness=2)
                    # cv2.putText(img,str(count), tuple(pt), font, 0.3, (255, 0, 0), 1)
                    count += 1
            # get average of light

            cv2.imshow("img", img)
            # cv2.imshow("mask", v)
            # cv2.imshow("mask2", v2)
            cv2.waitKey(0)
            yinyangScore[personNum] = rate
            personNum +=1
    drawHist(yinyangScore)
    cv2.waitKey(0)
    print "lightcount",lightcount




def demo_keypoint68_txt():
    from detect.dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    img_path = gb.glob(r"D:\data\face\uface_for_register\images\*g")
    # img_path = gb.glob(r"./image\*g")
    histLength = len(img_path)
    yinyangScore = np.zeros(histLength)
    personNum = 0
    with open("yinyang.txt","w") as fin:
        for path in img_path:
            img = cv2.imread(path)
            h, w = img.shape[:2]
            mask = np.zeros([h, w], dtype=np.uint8)
            mask2 = np.zeros([h, w], dtype=np.uint8)
            # for some reason some too big picture may could not find face,so the shape will devided
            shapes, rects = predictor(img)
            for shape in shapes:
                left = []
                for pt in shape[:9]:
                    left.append(pt)
                left.append(shape[28])
                left.append(shape[27])
                for i in range(17, 21):
                    left.append(shape[17 + 21 - i])
                leftface = np.array([[left]], dtype=np.int32)
                cv2.fillPoly(mask, leftface, 255)
                leftmask = cv2.bitwise_and(img, img, mask=mask)
                right = []
                for pt in shape[8:16]:
                    right.append(pt)
                right.append(shape[26])
                right.append(shape[24])
                right.append(shape[22])
                for pt in shape[27:30]:
                    right.append(pt)
                right = np.array([[right]], dtype=np.int32)
                cv2.fillPoly(mask2, right, 255)
                rightmask = cv2.bitwise_and(img, img, mask=mask2)
                leftmask_hsv = cv2.cvtColor(leftmask, cv2.COLOR_BGR2HSV)
                rightmask_hsv = cv2.cvtColor(rightmask, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(leftmask_hsv)
                h2, s2, v2 = cv2.split(rightmask_hsv)
                leftarea = np.sum(mask) / 255.
                rightarea = np.sum(mask2) / 255.
                leftvrate = np.sum(v) * 1.0 / leftarea
                rightRate = np.sum(v2) * 1.0 / rightarea
                # print leftarea, rightarea, "leftRate==", leftvrate, "rightRate==", rightRate
                minusrate = abs(leftvrate - rightRate)
                rate = abs(leftvrate / (leftvrate + rightRate) - 0.5)
                if personNum%100==0:
                    print personNum
                yinyangScore[personNum] = rate
                personNum += 1
                fin.write("%s_%.3f_%.3f\n"%(path,rate,minusrate))
    drawHist(yinyangScore)
    cv2.waitKey(0)

def demo_keypoint68_68txt():
    working_dir = r"D:\workspace\utils\drawPoints\test2/"
    with open(working_dir+'test.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            odom = line.split()[0].split("/")  # 将单个数据分隔开存好
            imgName = odom[-1]
            print working_dir+"/test2/"+ imgName
            img = cv2.imread(working_dir+"/test2/"+ imgName)
            point = line.split()
            shape_pts = []
            shapes = []
            for i in xrange(68):
                x1_temp = point[i + 1].split(",")[0].split(".")[0]
                y1_temp = point[i + 1].split(",")[1].split(".")[0]
                cv2.circle(img, ((int)(x1_temp), (int)(y1_temp)), 2, (0, 255, 255), thickness=2)
                shape_pts.append([(int)(x1_temp), (int)(y1_temp)])
            shapes.append(shape_pts)
            h, w = img.shape[:2]
            mask = np.zeros([h, w], dtype=np.uint8)
            mask2 = np.zeros([h, w], dtype=np.uint8)
            # for some reason some too big picture may could not find face,so the shape will devided
            for shape in shapes:
                left = []
                for pt in shape[:9]:
                    left.append(pt)
                left.append(shape[28])
                left.append(shape[27])
                for i in range(17, 21):
                    left.append(shape[17 + 21 - i])
                leftface = np.array([[left]], dtype=np.int32)
                cv2.fillPoly(mask, leftface, 255)
                leftmask = cv2.bitwise_and(img, img, mask=mask)
                right = []
                for pt in shape[8:16]:
                    right.append(pt)
                right.append(shape[26])
                right.append(shape[24])
                right.append(shape[22])
                for pt in shape[27:30]:
                    right.append(pt)
                right = np.array([[right]], dtype=np.int32)
                cv2.fillPoly(mask2, right, 255)
                rightmask = cv2.bitwise_and(img, img, mask=mask2)
                leftmask_hsv = cv2.cvtColor(leftmask, cv2.COLOR_BGR2HSV)
                rightmask_hsv = cv2.cvtColor(rightmask, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(leftmask_hsv)
                h2, s2, v2 = cv2.split(rightmask_hsv)
                leftarea = np.sum(mask) / 255.
                rightarea = np.sum(mask2) / 255.
                leftvrate = np.sum(v) * 1.0 / leftarea
                rightRate = np.sum(v2) * 1.0 / rightarea
                # print leftarea, rightarea, "leftRate==", leftvrate, "rightRate==", rightRate
                minusrate = abs(leftvrate - rightRate)
                rate = abs(leftvrate / (leftvrate + rightRate) - 0.5)
                print "minusrate = ",minusrate
                print rate,point[-1]
                cv2.imshow("mask", v)
                cv2.imshow("mask2", v2)
            cv2.imshow("img", img)
            cv2.waitKey(0)

def drawHist(hist):
    # s = np.random.rand(1, sampleNo )
    plt.subplot(111)
    plt.hist(hist, 20, normed=True)  #####bins=10
    plt.show()


if __name__=='__main__':
    # freeze_support()
    ###########---Loading detector modle---##############################################################################
    # detector = MtcnnDetector(model_folder='../model', ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
    # showmodel()
    # demo_keypointfive()
    demo_keypoint68()
    # demo_keypoint68_txt()
    # demo_keypoint68_68txt()
    # drawHist()