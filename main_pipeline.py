from __future__ import division
import cv2
import numpy as np
from image_warp import ImageWarper
from camera_calibration import CameraCalibration
from polynomial_fit import PolynomialFit
from thresholding import ApplyThreshold

cap = cv2.VideoCapture('videos/challenge.mp4')
cc = CameraCalibration('camera_cal/', rebuild_model=False)

fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')
_, sample_image = cap.read()
size = (1280, 720)
out = cv2.VideoWriter('Advanced_Lane_detector_2.mp4', fourcc, 25, size, True)

at = ApplyThreshold()
pt = ImageWarper()
fit_line = PolynomialFit(num_windows=9)

while cap.isOpened():
    ret, original_image = cap.read()

    if ret is True:
        # Undistort Image
        undist_image = cc.undistort_image(original_image)

        # Apply color, magnitude and direction threshols
        at.set_image(undist_image)
        mask, [d, m, c] = at.apply_threshold()
        masked_im = cv2.bitwise_or(undist_image, undist_image, mask=mask)

        # Apply perspective transform
        bird_view_mask = pt.get_bird_view(mask)

        # Fit the polynomial
        [points_right, points_left, info] = fit_line.poly_fit(bird_view_mask=bird_view_mask)
        # Overlay the lane markers
        final_image = fit_line.poly_fit_overlay(original_im=original_image, right_lane_point=points_right, left_lane_point=points_left, pt=pt)

        # Overlay Text
        font = cv2.FONT_HERSHEY_TRIPLEX
        curvature_text = "Lane Curvature: Left Lane = " + str(np.round(info[0], 2)) + ", Right = " + str(
            np.round(info[1], 2))
        cv2.putText(final_image, curvature_text, (30, 60), font, 1, (0, 255, 0), 2)
        deviation_text = "Lane deviation from center = {:.2f} m".format(info[2])
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(final_image, deviation_text, (30, 90), font, 1, (0, 255, 0), 2)
        out.write(final_image)
        # Display the frame
        # cv2.imshow('main window', fit_line.bird_view_im)
        # cv2.imshow('main window', final_image)
        # cv2.waitKey(10)
    else:
        break
        out.release()




