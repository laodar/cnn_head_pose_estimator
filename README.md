# CNN based head pose estimator

## About

    This is a python/mxnet implementation of a very simple CNN that predicts the head pose

    num of params < 100K

    CNN model:

    conv1 3*3*32,(2,2),relu

    conv2 3*3*32,(2,2),relu

    conv3 3*3*64,(2,2),relu

    conv4 3*3*64,(2,2),relu

    fc1   128,relu

    fc2   2,tanh

    dataset:

    1.http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html

    2.[Biwi Kinect Head Pose Database](http://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz)

## Requirement

-opencv

    only tested on 2.4.9.1

-mxnet

    only tested on 0.7.0

-mtcnn

    I use https://github.com/pangyupo/mxnet_mtcnn_face_detection to do face cropping and alignment

    padding = 0.27,desired_size = 64

## Test

run:

``python main.py``

examples from validation set:(green as label,red as prediction)
<p align="center">
<img src="examples_in_validation_set.jpg" width="960">
</p>

## Notice

so sorry for that my model is not robust in real scene, it seems to be sensitive to the background and lights because of the oversimplified public dataset,maybe we can synthesis more data with better background based on HPDatabase.
