import numpy as np
import mxnet as mx
import cv2

class HeadPoseEstimator(object):
    '''
    This is a CNN based head pose estimator in mxnet version
    CNN model:
    conv1 3*3*32,(2,2),relu
    conv2 3*3*32,(2,2),relu
    conv3 3*3*64,(2,2),relu
    conv4 3*3*64,(2,2),relu
    fc1   128,relu
    fc2   2,tanh
    '''
    def __init__(self,model_prefix='./model/cpt',ctx=mx.cpu()):
        '''
                Initialize the estimator
            Parameters:
            ----------
                model_prefix: string
                    path for the pretrained mxnet models
                ctx: context
                    the context where CNN running on
        '''
        self.model = mx.model.FeedForward.load(model_prefix, 1, ctx=ctx)

    def predict(self,img):
        '''
                Predict the pose on the cropped image
            Parameters:
            ----------
                img:a 64*64*3 numpy array,bgr order,range [0,255]
            Returns:
            -------
                [[pitch,yaw]] numpy array,range [-90,90]
        '''
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)[None,:,:,:]
        return self.model.predict(img.transpose([0, 3, 1, 2]))[0]

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat:

        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape:
            to_shape:
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0] / 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] / 2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self,img,points,desired_size=64,padding=0.27):
        """
                   crop and align face without rotation
               Parameters:
               ----------
                   img: numpy array, bgr order of shape (1, 3, n, m)
                       input image
                   points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                   desired_size: default 64
                   padding: default 0.27
               Returns:
               -------
                   crop_imgs: n*64*64*3 numpy array,rgb order
                       cropped and aligned faces without rotation
               """
        crop_imgs = []
        for p in points:
            shape = []
            for k in range(len(p) / 2):
                shape.append(p[k])
                shape.append(p[k + 5])

            if padding > 0:
                padding = padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape) / 2):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2 * i], shape[2 * i + 1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            #angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

            from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
            to_center = [0, 0]
            to_center[1] = desired_size * 0.4
            to_center[0] = desired_size * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), 0.0, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
            crop_imgs.append(cv2.cvtColor(chips,cv2.COLOR_BGR2RGB))

        return np.stack(crop_imgs)

    def crop_and_predict(self,img,points):
        '''
                Crop the detected faces in the image and predict the poses of them
            Parameters:
            ----------
                img:numpy array,a bgr order image containing faces
                points:numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                    n 5-points of face detected by mtcnn
                    https://github.com/pangyupo/mxnet_mtcnn_face_detection
            Returns:
            -------
                n*2 (pitch,yaw) numpy array,range [-90,90]
        '''
        imgs = self.extract_image_chips(img,points)
        return self.model.predict(imgs.transpose([0, 3, 1, 2]))[0]

