import numpy as np
import cv2
import dlib
import glob
import os

im_floder = "E:/Python/python_user/Practice_codes/second" \
            "/facenet-master/300w-lp-dataset/64_CASIA-FaceV5"
crop_im_path = "E:/Python/python_user/Practice_codes/second/" \
              "facenet-master/300w-lp-dataset/64_CASIA-FaceV5/crop_image_160"

#读取大文件夹下面所有500个小文件夹
im_floder_list = glob.glob(im_floder + "/*")

#通过dlib定义一个检测器
detector = dlib.get_frontal_face_detector()

idx = 0
#遍历每个小文件夹下所有图片
for idx_floder in im_floder_list:
    im_items_list = glob.glob(idx_floder  + "/*")

    #针对不同人创建不同的文件夹
    if not os.path.exists("{}/{}".format(crop_im_path, idx)):
        os.mkdir("{}/{}".format(crop_im_path, idx))

    idx_im = 0
    #遍历每张图片，预测人脸框的位置
    for im_path in im_items_list:
       # print(im_path)
        i_data = cv2.imread(im_path)
        im_data = cv2.cvtColor(i_data, cv2.COLOR_BGR2GRAY)

        dets = detector(im_data, 1)
        #打印人脸框
        print(dets)
        #判断是否检测到人脸框
        if dets.__len__() == 0:
            continue
        #如果取到人脸框，获取人脸框坐标
        d = dets[0]
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        #为了取到完整人脸，进行区域扩充,整体面积变大
        y1 = int(y1 - (y2 - y1) * 0.3)
        x1 = int(x1 - (x2 - x1) * 0.05)
        x2 = int(x2 + (x2 - x1) * 0.05)
        y2 = y2

        #裁剪
        im_crop_data = im_data[y1:y2, x1:x2]
        im_data = cv2.resize(im_crop_data, (160,160))
        im_save_path = "{}/{}/{}_{}.jpg".format(crop_im_path, idx, idx, "%04d" % idx_im)
        cv2.imwrite(im_save_path, im_data)
        idx_im += 1
    idx += 1