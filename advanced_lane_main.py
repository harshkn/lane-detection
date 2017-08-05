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
size = (1280, 900)
out = cv2.VideoWriter('Advanced_Lane_detector_3.mp4', fourcc, 25, size, True)

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
        bird_view_im = pt.get_bird_view(masked_im)

        # Fit the polynomial
        [points_right, points_left, info] = fit_line.poly_fit(bird_view_mask=bird_view_mask, bird_view_image=bird_view_im)
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

        # Display the frame
        # cv2.imshow('main window', fit_line.bird_view_im)

        sm_undistort = cv2.resize(undist_image, (320, 180))
        sm_mask_im = cv2.resize(masked_im, (319, 180))
        sm_bird_view_im = cv2.resize(bird_view_im, (320, 180))
        sm_bird_view_point = cv2.resize(fit_line.points_im_after, (320, 180))
        sm_normal_view = cv2.resize(fit_line.normal_point_im, (320, 180))

        deviation_text = "Undistored Image"
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(sm_undistort, "Undistored Image", (30, 30), font, 1, (0, 255, 0), 1)

        cv2.putText(sm_mask_im, "Color and Gradient", (30, 30), font, 1, (0, 255, 0), 1)
        cv2.putText(sm_mask_im, "Thresholding", (40, 50), font, 1, (0, 255, 0), 1)
        cv2.putText(sm_bird_view_point, "Perspective Transform", (30, 30), font, 1, (0, 255, 0), 1)
        cv2.putText(sm_bird_view_point, "and Line fitting", (40, 50), font, 1, (0, 255, 0), 1)
        cv2.putText(sm_normal_view, "Inverse Persp Transform", (20, 30), font, 1, (0, 255, 0), 1)

        all_im = np.hstack((sm_undistort, sm_mask_im, 255 * np.ones((180, 1, 3), dtype=np.uint8), sm_bird_view_point,
                            sm_normal_view))
        all_im = np.vstack((all_im, final_image))
        # print(type(sm_mask_im[0,0,0]))
        # print (all_im.shape)
        out.write(all_im)


        # cv2.imshow('main window', all_im)
        # cv2.waitKey(10)
    else:
        break
        # out.release()




