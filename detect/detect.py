import cv2
import numpy as np
import os
import time
from getList import list_image,read_list
from dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = cv_get_frontal_face_shape_detector(predictor_path)
detector = cv_get_frontal_face_detector()


def get_bbox(face_shape, bound):
    height, width = bound
    max_x, max_y = np.max(face_shape, axis=0)
    min_x, min_y = np.min(face_shape, axis=0)    
    if 0 <= min_x < max_x < width and 0 <= min_y < max_y < height:
        return min_x, min_y, max_x, max_y
    else:
        return None
def demo():
    img = cv2.imread("./data/2.jpg")
    print img.shape,img.shape[0],img.shape[1]
    # for some reason some too big picture may could not find face,so the shape will devided
    img = cv2.resize(img,(img.shape[1]/4,img.shape[0]/4))
    t0 = time.time()
    shape, rects = predictor(img)
    # draw point cloud
    draw_shapes(img,shape)
    if rects is not None:
        for rect in rects:
            x1,y1,x2,y2 = rect
            print rect
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.imwrite("2_test.jpg",img)
    cv2.imshow("img",img)
    cv2.waitKey(0)

def video_detect():
    cap = cv2.VideoCapture(0)
    count = 0
    while 1:
        ret, img = cap.read()
        if ret:
            # predict face shape and face area
            shape, rects = predictor(img)
            # draw point cloud
            draw_shapes(img, shape)
            # shape = np.array(shape).reshape((-1, 2))
            if rects is not None:
                for b in rects:
                    chip = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    if chip.size !=0:
                        print  len(chip),chip.size
                        cv2.rectangle(img,(int(b[0]), int(b[1])), (int(b[2]), int(b[3])),(0,255,0), 2)#alive green
                key = cv2.waitKey(1)
                if key == 27:
                    break
            else:
                print "This time have no face detected, please try it later ..."
                cv2.waitKey(1)
            cv2.imshow("img", img)
        else:
            print "device is not ready ..."
            cv2. waitKey(1)

def demo_saveList():
    img = cv2.imread("2.jpg")
    workdir = "./data/"
    listImage  = list_image(workdir,recursive=True)
    path_out = "hhe.list"
    with open(path_out, 'w') as fout:
        for i ,item,in enumerate(listImage):
            print i,item
            path = workdir+item[1]
            img = cv2.imread(path)
            print img.shape,img.shape[0],img.shape[1]
            # for some reason some too big picture may could not find face,so the shape will devided
            img = cv2.resize(img,(img.shape[1]/4,img.shape[0]/4))
            t0 = time.time()
            shape, rects = predictor(img)
            # draw_shapes(img,shape)
            if rects is not None:
                for rect in rects:
                    x1,y1,x2,y2 = rect
                    line = '%s ' % path
                    line += '%d %d %d %d \n' %(int(x1),int(y1),int(x2),int(y2))
                    fout.write(line)
            cv2.imwrite("2_test.jpg",img)
            cv2.imshow("img",img)
            # cv2.waitKey(0)
if __name__ == "__main__":
    video_detect()
