#读取300w-lp数据
from scipy.io import loadmat
import cv2

m = loadmat("E:/Python/python_user/Practice_codes"
            "/second/facenet-master/300w-lp-dataset/300W_LP/landmarks/AFW/AFW_134212_1_0_pts.mat")

landmark = m['pts_2d']

m_data = cv2.imread("E:/Python/python_user/Practice_codes"
            "/second/facenet-master/300w-lp-dataset/300W_LP/AFW/AFW_134212_1_0.jpg")

for i in range(68):
    cv2.circle(m_data, (int(landmark[i][0]), int(landmark[i][1])),
                        2, (0,255, 0), 2)
cv2.imshow("11", m_data)
cv2.waitKey(0)
