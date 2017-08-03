from __future__ import division
import numpy as np
import cv2
import os
import glob
from image_warp import ImageWarper
from camera_calibration import CameraCalibration
import matplotlib.pyplot as plt

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
    o_image = main_image.copy()

    # cv2.imwrite('orig.jpg', main_image)
    # Undistort Image
    cc = CameraCalibration('camera_cal/', rebuild_model=False)
    image = cc.undistort_image(o_image)
    # cv2.imwrite('undist.jpg', image)

    # Apply color, magnitude and direction threshols
    at = ApplyThreshold()
    at.set_image(image)
    mask, [d, m, c] = at.apply_threshold()
    masked_im = cv2.bitwise_or(image, image, mask=mask)

    # cv2.imwrite('color.jpg', 255 * c)
    # cv2.imwrite('dir.jpg', 255 * d)
    # cv2.imwrite('mag.jpg', 255 * m)

    # Apply perspective transform
    pt = ImageWarper()
    # print(pt.M)
    bird_view = pt.get_bird_view(mask)
    bird_view_image = pt.get_bird_view(masked_im)
    # cv2.imwrite('bv_mask.jpg', 255 * bird_view)
    # cv2.imwrite('bv_mask_im.jpg', bird_view_image)

    img = (bird_view).copy()

    print(img.shape)

    bv_copy = (255 * bird_view).copy()

    number_of_windows = 9  # Number of slices
    win_height = img.shape[0] / number_of_windows  # Height of each slice
    win_hist_left = []
    win_hist_right = []
    win_hist_full = []
    points_left_x = []
    points_right_x = []
    upd_points_left_y = []  # Final left side y
    upd_points_right_y = []  # Final right side y

    # Compute histogram for each window and get left and right lane markers using argmax
    for window in range(number_of_windows):
        # Split the window into small slices
        win_start = int(window * win_height)
        win_end = int((window + 1) * win_height)

        # Compute full histogram for each slices
        full_hist_entry = np.sum(img[win_start:win_end, :], axis=0)
        win_hist_full.append(full_hist_entry)

        # Store the y coordinates for each slices
        upd_points_left_y.append(int(win_end))
        upd_points_right_y.append(int(win_end))

        # Compute left half histogram
        hist_entry = np.sum(img[win_start:win_end, :int(img.shape[1]/2)], axis=0)
        # Compute the max point in the left histogram and that is the x point
        hist_max_left = np.argmax(hist_entry)
        win_hist_left.append(hist_entry)
        points_left_x.append(hist_max_left)

        # Compute right half histogram
        hist_entry = np.sum(img[win_start:win_end, int(img.shape[1] / 2):], axis=0)
        # Compute the max point in the right histogram and that is the x point
        hist_max_right = np.argmax(hist_entry)
        points_right_x.append(hist_max_right + int(img.shape[1] / 2))
        win_hist_right.append(hist_entry)

    init_hist_max_left = points_left_x[-1]
    init_hist_max_right = points_right_x[-1]
    error_th = 150
    upd_points_left_x = len(points_left_x) * [None]  # Final left side x
    upd_points_right_x = len(points_right_x) * [None]  # Final right side x

    # Plot the lane marker estimate on the hist plot
    for ind, hist_data in reversed(list(enumerate(win_hist_full))):
        # plt.subplot(number_of_windows, 1, ind+1)
        # plt.xticks([]), plt.yticks([])
        # plt.plot(hist_data)
        # update the lane markers with previous slice data if lane data does not exist
        # for that slice
        if abs(init_hist_max_left - points_left_x[ind]) > error_th:
            upd_points_left_x[ind] = init_hist_max_left
        else:
            upd_points_left_x[ind] = (points_left_x[ind])
            init_hist_max_left = points_left_x[ind]

        # update the lane markers with previous slice data if lane data does not exist
        # for that slice
        if abs(init_hist_max_right - points_right_x[ind]) > error_th:
            upd_points_right_x[ind] = init_hist_max_right
        else:
            upd_points_right_x[ind] = (points_right_x[ind])
            init_hist_max_right = points_right_x[ind]

        # plt.plot(100 * [upd_points_right_x[ind]], np.arange(0,100))
        # plt.plot(100 * [upd_points_left_x[ind]], np.arange(0, 100))

    # plt.savefig('full_hist.jpg')

    # for ind, hist_data in enumerate(win_hist_left):
    #     plt.subplot(number_of_windows, 1, ind+1)
    #     plt.xticks([]), plt.yticks([])
    #     plt.plot(hist_data)
    # plt.savefig('left_hist.jpg')

    for ind, hist_data in enumerate(win_hist_right):
        plt.subplot(number_of_windows, 1, ind+1)
        plt.xticks([]), plt.yticks([])
        plt.plot(hist_data)
    # plt.show()
    # plt.savefig('right_hist.jpg')

    # Annotate the points on the image
    for ind, _ in reversed(list(enumerate(upd_points_right_x))):
        cv2.circle(bird_view_image, (upd_points_right_x[ind], upd_points_right_y[ind]), 3, [0, 255, 0], -1)
        cv2.circle(bird_view_image, (upd_points_left_x[ind], upd_points_left_y[ind]), 3, [0, 255, 0], -1)

    cv2.imwrite('overlay_on_im.jpg', bird_view_image)

    # Polyfit the points
    # Right lane
    # Get 2nd degree polynomial
    poly_nom = np.polyfit(upd_points_right_y, upd_points_right_x, deg=2)
    lane_polynomial = np.poly1d(poly_nom)
    print(lane_polynomial)

    # Compute start and end values of y axis
    start_y = upd_points_right_y[0]
    end_y = upd_points_right_y[-1]
    # Get series of y values
    y_val_right = np.arange(start_y, end_y, 1)

    # Compute x values using the fitted polynomial
    x_val = lane_polynomial(y_val_right)
    x_val_right = [int(elem) for elem in x_val]  # Cast to int

    # ======Left lane=================
    # Get 2nd degree polynomial
    poly_nom = np.polyfit(upd_points_left_y, upd_points_left_x, deg=2)
    lane_polynomial = np.poly1d(poly_nom)
    print(lane_polynomial)

    # Compute start and end values of y axis
    start_y = upd_points_left_y[0]
    end_y = upd_points_left_y[-1]
    # Get series of y values
    y_val_left = np.arange(start_y, end_y, 1)

    # Compute x values using the fitted polynomial
    x_val = lane_polynomial(y_val_left)
    x_val_left = [int(elem) for elem in x_val]  # Cast to int

    for ind, _ in enumerate(x_val):
        cv2.circle(bird_view_image, (x_val_left[ind], y_val_left[ind]), 2, [255, 255, 0], -1)
    for ind, _ in enumerate(x_val):
        cv2.circle(bird_view_image, (x_val_right[ind], y_val_right[ind]), 2, [255, 255, 0], -1)


    normal_view_img = pt.get_normal_view(bird_view_image)
    norm_right = pt.get_normal_view_points(x_val_right, list(y_val_right))
    norm_left = pt.get_normal_view_points(x_val_left, list(y_val_left))

    blank_img = np.zeros_like(o_image).astype(np.uint8)

    for ind, _ in enumerate(norm_right):
        cv2.circle(blank_img, (norm_right[ind]), 4, [0, 0, 255], -1)
        cv2.circle(blank_img, (norm_left[ind]), 4, [0, 0, 255], -1)


    rev_norm_left = list(reversed(norm_left))
    full_poly = np.array([norm_right + rev_norm_left + [norm_right[0]]], dtype=np.int32)
    # full_poly = full_poly.append(norm_right[0])
    a3 = np.array([[[10, 10], [100, 10], [100, 100], [10, 100]]], dtype=np.int32)

    cv2.fillPoly(blank_img, full_poly, (0, 255, 0))
    print(norm_right[0], norm_right[-1])
    print(norm_left[0], norm_left[-1])


    result = cv2.addWeighted(o_image, 1, blank_img, 0.3, 0)
    # cv2.imwrite('final.jpg', result)
    # win_w =

    # cv2.imshow('win_2', result)
    # cv2.waitKey(0)




