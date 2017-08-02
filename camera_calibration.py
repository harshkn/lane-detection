import cv2
import os
import glob
import numpy as np


class CameraCalibration():

    def __init__(self, calib_folder, rebuild_model=False):
        self.full_path = os.path.join(os.getcwd(), calib_folder)
        self.images = glob.glob(self.full_path + '/*.jpg')
        (self.x_cor, self.y_cor) = (9, 6)
        self.objpoints = []
        self.imgpoints = []
        self.shape = None

        if rebuild_model is True:
            self.corner_points = []
            self.object_points = []
            self.shape = None
            self.find_corners()
            # ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            #     cv2.calibrateCamera(self.object_points, self.corner_points, self.shape, None, None)
        else:
            try:
                self.object_points = np.load('objpoints.npy')
                self.corner_points = np.load('imgpoints.npy')
                self.shape = tuple(np.load('imgshape.npy'))
            except:
                self.corner_points = []
                self.object_points = []
                self.shape = None
                self.find_corners()

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.object_points, self.corner_points, self.shape, None, None)



    def find_corners(self):
        '''
        Find the corners in the checkboard pattern
        :param image: 
        :return: 
        '''

        for each_image in self.images:
            image = cv2.imread(each_image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            base_objp = np.zeros((6 * 9, 3), np.float32)
            base_objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            self.shape = image_gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(image=image_gray,patternSize=(self.x_cor, self.y_cor))
            print(ret)
            if ret is True:
                # print(corners)
                self.corner_points.append(corners)
                self.object_points.append(base_objp)

        np.save('objpoints', self.object_points)
        np.save('imgpoints', self.corner_points)
        np.save('imgshape', self.shape)

    def undistort_image(self, image):
        undistorted_image = cv2.undistort(image, cameraMatrix=self.mtx, distCoeffs=self.dist,
                                          dst=None, newCameraMatrix=None)
        return undistorted_image


if __name__ == '__main__':
    cc = CameraCalibration('camera_cal/', rebuild_model=False)
    img = cc.images[2]
    image = cv2.imread(img)
    u_image = cc.undistort_image(image)
    # cv2.imshow('orif', image)
    # cv2.imshow('undistorted', u_image)
    # cv2.waitKey(0)




