import os
import cPickle as pkl
import cv2
import numpy as np
def list_image(root, recursive):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                if fname.endswith(".jpg") or fname.endswith(".bmp"):
                    if(i%1000==0):
                        print i
                    fpath = os.path.join(path, fname)
                    if os.path.isfile(fpath):
                        if path not in cat:
                            cat[path] = len(cat)
                        yield (i, os.path.relpath(fpath, root), cat[path])
                        i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

if __name__ == '__main__':
    # fid = open("img.pkl","rb")
    # container = pkl.load(fid)
    # print container.__len__()
    # for i,value in enumerate(container):
    #     print i,value
    from detect.dlib_face import cv_get_frontal_face_shape_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    working_dir = r"Y:\maxiaofang\data\10000\tight"
    # working_dir = r"./image/"
    imgList = list_image(working_dir,recursive=True)
    imgList = list(imgList)
    container = []
    count = 0
    for i in xrange(len(imgList)):
        imgName = str(imgList[i][1])
        path = os.path.join( working_dir,imgName)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        maskw = np.zeros([h, w], dtype=np.uint8)
        det_result = predictor(img)
        shapes, rects = det_result
        newContainerPart={}
        for shape in shapes:
            if True:
                count+=1
                if count%100==0:
                    print "lightcount = ",count
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
                detectResult =[det_result,res]
                newContainerPart[path] = detectResult
                # print mean_b,mean_g,mean_r
                # print "wholemask_v= ",mean_v,mean_v*255
                # cv2.imshow("wholemask",wholemask)
                # cv2.imshow("wholemask_hsv",wholemask_hsv)
                info = "%.f_%.f_%.f_%d.jpg"%(mean_b,mean_g,mean_r,count)
                avr_light = (mean_b+mean_g+mean_r)/3.
                min_light = min(mean_b,mean_g,mean_r)
                # cv2.imwrite(os.path.join(newPath,info),img)
                container.append(newContainerPart)

    outPut = open("img1W.pkl","wb")
    pkl.dump(container,outPut,protocol=2)
    outPut.close()