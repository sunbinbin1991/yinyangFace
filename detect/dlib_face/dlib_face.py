"""
                          -@o      +}                       
                         #dM       BW                       
                        $#W        ha                       
                         W         ?)@                      
                       &W           W+}                     
                      8#v           ?#8                     
                     &m& +L#88m      8b                     
                   M%h          +    ~ZBM                   
           IamGK*                  +      @#&WIamGK         
            8  B                            8x   &          
         +  & M                            + (k o+          
            #                                  8&           
           W                       +            8           
        + 8          +                           W}         
         8  ~ &%%%B88fdb&W           d&&kW&B&[u+  %         
         #  M8 8            b % 8_                 B        
         #                                   %W #  M        
    B%   8~                                      Bo@        
       @8 &jm                                   W&+         
         @8 m  ~                               &W 8         
       %c     /MM                           j8   #          
   +             ~& #8&&8&8&#&M&#Wm#&&M&8*~      ]          
          ~          W8           M&                        
                   8 8     r]_     &#                       
                  #  &   #     M   *}8                      
                 *   8  8-      M   M o                     
                 8+ ML  M_#W%Mu+%   W B                     
                  W%&   qbbWBB/W    #0                      
                   818  kQoo%h8#   8                        
                                %&W                          
"""

import dlib
import cv2
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def face_detection_video():
    detector = dlib.get_frontal_face_detector()   
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    cap = cv2.VideoCapture(0)
    show = 1
    while 1:
        time0 = time.time()
        ret, img0_bgr = cap.read()
        if ret:
            img0_rgb = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
            dets = detector(img0_rgb)
            
            if show:
                for i, d in enumerate(dets):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))    
                    cv2.rectangle(img0_bgr, (d.left(), d.top()), (d.right(), d.bottom()), (0,0,255))
                    shape = predictor(img0_rgb, d)
                    for p in range(shape.num_parts):
                        cv2.circle(img0_bgr, (shape.part(p).x, shape.part(p).y), 2, (255,0,0), thickness=2)
                    
                cv2.imshow("img0_bgr", img0_bgr)
                cv2.waitKey(1)          
        else:
            print "cannot open device0"
            
        print "Frame Rate", 1. / (time.time() - time0)
        
def cv_get_frontal_face_detector():
    detector = dlib.get_frontal_face_detector()
    def detect(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img_rgb)
        rets = []
        for d in dets:
            rets.append((d.left(), d.top(), d.right(), d.bottom()))
        return rets
    return detect

def cv_get_frontal_face_shape_detector(predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    def detect_shape(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img_rgb)
        rects = []
        shapes = []
        for d in dets:
            time0 = time.time()
            shape = predictor(img_rgb, d)
            #print "face shape regression time used", time.time() - time0
            rects.append((d.left(), d.top(), d.right(), d.bottom()))
            shape_pts = []
            for n in range(shape.num_parts):
                shape_pts.append([shape.part(n).x, shape.part(n).y])
            shapes.append(shape_pts)
        return shapes, rects
    return detect_shape

def detector_test():
    filename = "test.jpg"
    img = cv2.imread(filename)
    detector = cv_get_frontal_face_detector()
    rects = detector(img)
    print rects
    
    predictor_path = "model/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    shapes = predictor(img)
    print shapes
    
def point_cloud():
    def randrange(n, vmin, vmax):
        return (vmax - vmin)*np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zl, zh)
        ax.scatter(xs, ys, zs, c=c, marker=m)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
def draw_shapes(img, shapes):
    for shape in shapes:
        #color = tuple(np.random.randint(0,256,size=3))
        color = (255, 64, 128)
        #count = 0
        for pt in shape:
            cv2.circle(img, tuple(pt), 2, color, thickness=2)
            #print count
            #cv2.imshow("tmp", img)
            #cv2.waitKey(0)
            #count += 1