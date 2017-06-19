import mxnet as mx
import cv2
import numpy as np
from HeadPoseEstimator import HeadPoseEstimator

estimator = HeadPoseEstimator(model_prefix='./model/cpt',ctx=mx.cpu())

img = cv2.imread('test1.png')

#center crop to 64*64
img = img[5:74-5,5:74-5,:]

print 'yaw-pitch angle of test1:',estimator.predict(img)

img = cv2.imread('test2.jpg')

#you can use mtcnn to get the points,https://github.com/pangyupo/mxnet_mtcnn_face_detection
points = np.array([[ 80,103,80,109,125,179,166,220,236,235]])

print 'yaw-picth angle of test2:',estimator.crop_and_predict(img,points)