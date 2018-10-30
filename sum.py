import numpy as np
from multiprocessing import Process, freeze_support
from collections import namedtuple
from mtcnn_detector import MtcnnDetector
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import glob as gb
img_path = gb.glob("../image/*.jpg")
path_in = r"D:\test\0804\detectScore.txt"
aliveNum = 0
spoofNum = 0
eyenopass = 0
eyepass = 0
facepass = 0
with open(path_in) as fin:
    while True:
        line = fin.readline()
        if not line:
            print aliveNum,spoofNum,"eyenopass=",eyenopass,"eyepass=",eyepass,",facepass=",facepass
            break
        result = line.split("\t")[-1]
        # print result
        flag = result.split(":")[-1].split("\n")[0]
        eyeresult = result.split(":")[1].split("(")[0]
        leftresult = result.split(":")[1].split("(")[1].split("_")[0]
        rigthtresult = result.split(":")[1].split("(")[1].split("_")[1].split(")")[0]
        faceresult = result.split("_")[2].split(":")[1].split("(")[0]
        if flag =="1":
            aliveNum +=1
        if flag =="0":
            spoofNum +=1
            if faceresult=="1":
                facepass+=1
            if eyeresult=="0":
                print leftresult,rigthtresult
                if (float)(leftresult)<0.5 and (float)(rigthtresult)<0.5:
                    eyenopass+=1
                if (float)(leftresult)>0.5 and (float)(rigthtresult)>0.5:
                  pass
                a = leftresult.split(".")[0]
                b = rigthtresult.split(".")[0]
                if leftresult=="0.00" and rigthtresult=="0.00":
                    eyepass += 1
                    print result
            # print "eye",eyeresult
