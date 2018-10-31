import cv2
import numpy as np
import os
from dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
import fire

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = cv_get_frontal_face_shape_detector(predictor_path)
detector = cv_get_frontal_face_detector()

folders = ["LiveSubjectsImages",
           "SpoofSubjectImages/MacBook_FrontCamera",
           "SpoofSubjectImages/MacBook_RearCamera",
           "SpoofSubjectImages/Nexus_FrontCamera",
           "SpoofSubjectImages/Nexus_RearCamera",
           "SpoofSubjectImages/PrintedPhoto_FrontCamera",
           "SpoofSubjectImages/PrintedPhoto_RearCamera",
           "SpoofSubjectImages/Tablet_FrontCamera",
           "SpoofSubjectImages/Tablet_RearCamera",]

filenames = os.listdir(folders[0])


def get_bbox(face_shape, bound):
    height, width = bound
    max_x, max_y = np.max(face_shape, axis=0)
    min_x, min_y = np.min(face_shape, axis=0)    
    if 0 <= min_x < max_x < width and 0 <= min_y < max_y < height:
        return min_x, min_y, max_x, max_y
    else:
        return None
    

class Preprocessing(object):    
    
    def extraction(self, anno="annotation.txt"):
        with open(anno, "wb") as f:
            cnt = 0
            for filename in filenames:
                for folder in folders:
                    cnt += 1
                    if cnt % 100 == 0:
                        print "processing", cnt
                    fullpath = os.path.join(folder, filename)
                    #print fullpath
                    img = cv2.imread(fullpath)
                    shapes, rects = predictor(img)
                    if len(shapes) == 1:
                        f.write(fullpath+" ")
                        for pt in shapes[0]:
                            f.write("%d %d "%(pt[0], pt[1]))
                        f.write("\n")   
                    else:
                        print fullpath, "number of faces", len(shapes)
                        #draw_shapes(img, shapes)
                        #cv2.imshow("img", img)
                        #cv2.waitKey(0)
                        #cv2.destroyWindow("img")
        f.close()
        
    def generate_chip(self, size=None, anno="annotation.txt"):
        cnt = 0
        dirname = str(size) if size else "orisize"
        out_dir = os.path.join("clip", dirname)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(anno) as f:
            for line in f:
                cnt += 1
                if cnt % 100 == 0:
                    print "processing", cnt
                raw = line.strip().split()
                img = cv2.imread(raw[0])
                shape = map(int, raw[1:])
                shape = np.array(shape).reshape((-1, 2))
                bbox = get_bbox(shape, img.shape[:2])
                if bbox:
                    x1, y1, x2, y2 = bbox
                    chip = img[y1:y2, x1:x2]
                    if size:
                        chip = cv2.resize(chip, (size, size))
                    #cv2.imshow("chip", chip)
                    #cv2.waitKey(0)
                    fname = ".".join(raw[0].replace("\\", '/').split('/'))
                    #print fname
                    out_path = os.path.join(out_dir, fname)
                    cv2.imwrite(out_path, chip)
                else:
                    print raw[0]


if __name__ == "__main__":
    fire.Fire(Preprocessing)
            