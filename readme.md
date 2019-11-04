##### 图片质量估计-如何判定一个人脸是否为阴阳脸

##### 前言：
在人脸识别中，人脸质量对人脸识别系统重要性不言而喻。本文主要简要说明，如何进行阴阳脸的检测。思路还是挺清晰的：获取人脸左右部分的亮度进行比较，差异较大则为阴阳脸，否则则认为是正常人脸。其实最开始的时候，考虑这个问题的时候，希望用整个区域的人脸亮度的方差值作为作为一个衡量标准，但是在实际测试的时候发现，这个区分度并不明显。故而选择安装人脸的左右区分来划分阴阳脸。

----
 废话不多说，直接上干货：依赖dlib检测人脸，可以使用如下的dlib进行检测人脸关键点，觉得还行的，给个star行不行。。。毕竟今天是七夕，我还在开心的敲博客。。。心疼自己一秒，链接如下：
 [https://github.com/sunbinbin1991/detect](https://github.com/sunbinbin1991/detect)，下面是具体判定的方法，主要是如何获取所需的人脸区域：配合上面链接服用，效果更佳。
```python
def demo_checkyinyang():
    from detect.dlib_face import cv_get_frontal_face_shape_detector, draw_shapes, cv_get_frontal_face_detector
    predictor_path = "./detect/models/shape_predictor_68_face_landmarks.dat"
    predictor = cv_get_frontal_face_shape_detector(predictor_path)
    detector = cv_get_frontal_face_detector()
    # img = cv2.imread("../image/3.jpg")
    img_path = gb.glob("./image/*g")#获取指定文件夹下的所有人脸照片
    for path in img_path:
        img = cv2.imread(path)
        h,w = img.shape[:2]
        mask =  np.zeros([h, w], dtype = np.uint8)
        mask2 =  np.zeros([h, w], dtype = np.uint8)
        shapes, rects = predictor(img)#获取检测人脸关键点和人脸框
        # draw point cloud
        for shape in shapes:
            count = 0
            for pt in shape:
                # cv2.circle(img, tuple(pt), 2, color, thickness=2)
                # cv2.putText(img,str(count), tuple(pt), font, 0.3, (255, 0, 0), 1)
                count+=1
        for shape in shapes:
            keypoint_68= []
            x1,y1 = np.min(shape, axis=0)
            x2,y2 = np.max(shape, axis=0)
            # part1 ：使用方差来判定，效果不行
            mean_pixels = (127.5)
            # _mean_pixels = np.array(mean_pixels).reshape((1, 2))
            img_var = img[y1:y2,x1:x2].copy()
            img_var = cv2.cvtColor(img_var, cv2.COLOR_BGR2HSV)
            img_varH, img_varS, img_varV = cv2.split(img_var)
            data = img_varV.astype('float32')
            _data = (data - np.array(mean_pixels)) * 1 / 127.5
            cv2.imshow("var",img[y1:y2,x1:x2])
            print "np.var = ",np.var(_data)
            # part2： 使用左右脸来进行区分
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
            right = np.array([[right]],dtype=np.int32)
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
            rate = abs(leftvrate / (leftvrate + rightRate) - 0.5)
            print rate
            cv2.imshow("img", img)
            cv2.imshow("mask", leftmask)
            cv2.imshow("mask2", rightmask)
            cv2.waitKey(0)
```
* 结果如下：
* 原始图片：![](https://imgconvert.csdnimg.cn/aHR0cDovL3d3MS5zaW5haW1nLmNuL2xhcmdlLzAwNU0ybWFDZ3kxZnVkNTdmYWVtOGozMDR5MDYycTMxLmpwZw?x-oss-process=image/format,png)
* 左脸区域：![](https://imgconvert.csdnimg.cn/aHR0cDovL3d3MS5zaW5haW1nLmNuL2xhcmdlLzAwNU0ybWFDZ3kxZnVkNTd3bmd3OWozMDR5MDYyNzQ0LmpwZw?x-oss-process=image/format,png)
* 右脸区域：![](https://imgconvert.csdnimg.cn/aHR0cDovL3d3MS5zaW5haW1nLmNuL2xhcmdlLzAwNU0ybWFDZ3kxZnVkNTg3ZjlvcmozMDR5MDYyanI3LmpwZw?x-oss-process=image/format,png)


---

后续：最后只要将通道分离，获取V通道就可以计算亮度差异值，然后用作阴阳脸的判定。接下来将有一篇博客来实现C++版本的检测。
#### TO BE CONTINUE...