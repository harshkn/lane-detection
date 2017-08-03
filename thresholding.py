from __future__ import division
import numpy as np
import cv2
import os
import glob
from image_warp import ImageWarper
from camera_calibration import CameraCalibration
from polynomial_fit import PolynomialFit

class ApplyThreshold:

    def __init__(self):
        self.img = None
        self.sobel_ksize = 3
        self.sob_dir = {'min': 0.75, 'max': 1.2}
        self.sob_mag = {'min': 50, 'max': 255}

    def set_image(self, img):
        self.img = img

    def apply_direction_threshold(self, sobelx, sobely):
        # get gradient
        gradient_dir = np.arctan2(np.abs(sobelx), np.abs(sobely))
        # init binary mask
        binary_mask_dir = np.zeros(gradient_dir.shape, dtype=np.uint8)
        # x = np.uint8(m * x / np.max(x))
        # populate the binary mask
        binary_mask_dir[(self.sob_dir['max'] > gradient_dir) & (gradient_dir > self.sob_dir['min'])] = 1
        return binary_mask_dir

    def apply_mag_threshold(self, sobelx, sobely):
        # get magnitude
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        magnitude = np.uint8(255*magnitude/np.max(magnitude))
        # init binary mask
        binary_mask_mag = np.zeros(magnitude.shape, dtype=np.uint8)
        # populate the binary mask
        binary_mask_mag[(self.sob_mag['max'] >= magnitude) & (magnitude >= self.sob_mag['min'])] = 1
        return binary_mask_mag

    def apply_color_threshold(self):
        # convert to hsv
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # yellow threshold
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        # white threshold
        lower_white = np.array([0, 0, 230], dtype=np.uint8)
        upper_white = np.array([255, 20, 255], dtype=np.uint8)
        # create mask
        yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
        # merge mask
        mask = np.zeros(yellow_mask.shape, dtype=np.uint8)
        mask[(white_mask == 255) | (yellow_mask == 255)] = 1
        # image_masked = cv2.bitwise_or(img_color, img_color, mask=mask)
        return mask

    def apply_threshold(self):
        gray_im = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_im, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        sobely = cv2.Sobel(gray_im, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        dir_mask = self.apply_direction_threshold(sobelx, sobely)
        mag_mask = self.apply_mag_threshold(sobelx, sobely)
        color_mask = self.apply_color_threshold()

        # merge different masks
        final_mask = np.zeros_like(dir_mask)
        final_mask[(((dir_mask == 1) & (mag_mask == 1)) | (color_mask == 1))] = 1
        # masked_im = cv2.bitwise_or(self.img, self.img, mask=dir_mask)
        return final_mask, [dir_mask, mag_mask, color_mask]

if __name__ == '__main__':

    # Get Image
    cp = os.getcwd()
    path = 'test_images'
    images = glob.glob(os.path.join(cp, path) + '/*.jpg')
    main_image = cv2.imread(images[3])
    original_image = main_image.copy()


    cc = CameraCalibration('camera_cal/', rebuild_model=False)
    at = ApplyThreshold()
    pt = ImageWarper()
    fit_line = PolynomialFit(num_windows=9)


    # Undistort Image
    undist_image = cc.undistort_image(original_image)


    # Apply color, magnitude and direction threshols
    at.set_image(undist_image)
    mask, [d, m, c] = at.apply_threshold()
    masked_im = cv2.bitwise_or(undist_image, undist_image, mask=mask)


    # Apply perspective transform
    bird_view_mask = pt.get_bird_view(mask)
    # bird_view_image = pt.get_bird_view(masked_im)

    [points_right, points_left] = fit_line.poly_fit(bird_view_mask=bird_view_mask)
    final_image = fit_line.poly_fit_overlay(original_im=original_image, right_lane_point=points_right, left_lane_point=points_left, pt=pt)


    cv2.imshow('win_2', final_image)
    cv2.waitKey(0)




