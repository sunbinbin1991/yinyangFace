#coding=utf-8
import cv2
import mxnet as mx
import numpy as np
import cPickle as pkl
import os
import random
from multiprocessing import Process, freeze_support
from collections import namedtuple
from mtcnn_detector import MtcnnDetector
import matplotlib.pyplot as plt
font = cv2.FONT_HERSHEY_SIMPLEX

import glob as gb
color = (255, 64, 128)

def demo():
    from detect.dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    detector = cv_get_frontal_face_detector()
    # img = cv2.imread("../image/3.jpg")
    # img_path = gb.glob(r"Y:\maxiaofang\data\Uface_register_face\*g")
    # img_path = gb.glob(r"Y:\maxiaofang\data\10000\tight\suc\*g")
    img_path = gb.glob(r"./reg/0.2-0.8/*g")
    histLength = len(img_path)
    for path in img_path:
        print path
        img = cv2.imread(path)
        img = cv2.imread(r"./image/test-12.jpg")
        h,w = img.shape[:2]
        mask =  np.ones([h, w,3], dtype = np.uint8)
        # for some reason some too big picture may could not find face,so the shape will devided
        shapes, rects = predictor(img)
        for shape in shapes:
            keypoint_68= []
            x1,y1 = np.min(shape, axis=0)
            x2,y2 = np.max(shape, axis=0)
            mean_pixels = (127.5)
            # _mean_pixels = np.array(mean_pixels).reshape((1, 2))
            img_var = img[y1:y2,x1:x2].copy()
            img_var = cv2.cvtColor(img_var, cv2.COLOR_BGR2HSV)
            img_varH, img_varS, img_varV = cv2.split(img_var)
            chip = img[y1:y2,x1:x2]

            # mask_t = np.multiply(mask,1)
            # mask_tmp =
            # cv2.imshow("chip",chip)
            img = img.astype("float32")
            mask_t = mask.astype("float32")
            mask_t[y1:y2, x1:x2] = 0.5
            img_t = np.multiply(img,mask_t)
            img_t = img_t.astype("uint8")
            # hsv = cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)  # convert it to hsv
            # h, s, v = cv2.split(hsv)
            # v += 25
            # final_hsv = cv2.merge((h, s, v))
            # chip_r = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            # cv2.imwrite("image_processed.jpg", img)
            # cv2.imshow("chip_r",chip_r)
            cv2.imshow("img",img)
            cv2.imshow("mask",mask_t)
            cv2.imshow("img_t",img_t)
            cv2.waitKey(0)

def demo_checkill():
    from detect.dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    # img_path = gb.glob(r"Y:/maxiaofang/data/Uface_register_face/*g")
    img_path = "D:/data/tt/reg/"
    personNum = 0
    lightcount =1
    for path in img_path:
        # print path
        img = cv2.imread(path)
        img = cv2.imread(r"./image/0.93_267.jpg")
        h,w = img.shape[:2]
        maskw =  np.zeros([h, w], dtype = np.uint8)
        # for some reason some too big picture may could not find face,so the shape will devided
        shapes, rects = predictor(img)
        for shape in shapes:
            if True:
                lightcount+=1
                if lightcount%100==0:
                    print "lightcount = ",lightcount
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
                # print "wholemask_v= ",mean_v,mean_v*255
                lightcount+=1
                # cv2.imshow("wholemask",wholemask)
                # cv2.imshow("wholemask_hsv",wholemask_hsv)
                info = "%.f_%.f_%.f.jpg"%(mean_b,mean_g,mean_r)
                min_light = min(mean_b,mean_g,mean_r)
                # if min_light<50:
                #     newPath = os.path.join(save_dir,"dark")
                # elif min_light>200:
                #     newPath = os.path.join(save_dir,"light")
                # else:
                #     continue
                # if (not os.path.exists(newPath)):
                #     os.mkdir(newPath)
                # cv2.imwrite(os.path.join(newPath,info),img)

            cv2.imshow("img", img)
            cv2.waitKey(0)
            personNum +=1
    # cv2.waitKey(0)
    print "lightcount",lightcount


def checkPKl():
    fid = open("img.pkl","rb")
    container = pkl.load(fid)
    print container.__len__()
    lightcount = 0
    save_dir = "D:/data/tt/reg/1026"
    random.shuffle(container)
    newContainer = []
    for i,value in enumerate(container):
        imgPath = value.keys()[0]
        detectResult = value[imgPath]
        img = cv2.imread(imgPath)
        h, w = img.shape[:2]
        maskw = np.zeros([h, w], dtype=np.uint8)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # print i,value
        newContainerPart= {}
        shapes, rects = detectResult
        for shape in shapes:
            if True:
                lightcount+=1
                if lightcount%100==0:
                    print "lightcount = ",lightcount
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
                res = [mean_b,mean_g,mean_r]
                detectResult =[detectResult,res]
                newContainerPart[imgPath] = detectResult
                # print mean_b,mean_g,mean_r
                # print "wholemask_v= ",mean_v,mean_v*255
                # cv2.imshow("wholemask",wholemask)
                # cv2.imshow("wholemask_hsv",wholemask_hsv)
                info = "%.f_%.f_%.f_%d.jpg"%(mean_b,mean_g,mean_r,lightcount)
                avr_light = (mean_b+mean_g+mean_r)/3.
                min_light = min(mean_b,mean_g,mean_r)
                if avr_light<50:
                    newPath = os.path.join(save_dir,"dark")
                elif min_light>220:
                    newPath = os.path.join(save_dir,"light")
                elif (min_light>150 and min_light<30):
                    newPath = os.path.join(save_dir, "unNormalColor")
                else:
                    newPath = os.path.join(save_dir, "Normal")
                if (not os.path.exists(newPath)):
                    os.mkdir(newPath)
                # cv2.imwrite(os.path.join(newPath,info),img)
                newContainer.append(newContainerPart)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

    # outPut = open("imgRGB.pkl","wb")
    # pkl.dump(newContainer,outPut,protocol=2)
    # outPut.close()

checkPKl()
# demo_checkill()