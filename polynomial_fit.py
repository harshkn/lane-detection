from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np

class PolynomialFit:

    def __init__(self, num_windows=9):
        # self.original_im = im
        self.bird_view_im = None
        self.points_im_before = None
        self.points_im_after = None
        self.normal_point_im = None
        self.num_of_windows = num_windows

    def poly_fit(self, bird_view_mask, bird_view_image):

        self.bird_view_im = bird_view_mask
        # img = self.bird_view_im.copy()
        img = bird_view_mask
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
            hist_entry = np.sum(img[win_start:win_end, :int(img.shape[1] / 2)], axis=0)
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

        init_hist_max_left = int(img.shape[1] / 4)
        elem_greater = list(it for it in points_right_x if it > int(0.7 * img.shape[1]))
        try:
            init_hist_max_right = np.min(elem_greater)
        except:
            init_hist_max_right = int(0.85 * img.shape[1])

        error_th = 140
        upd_points_left_x = len(points_left_x) * [None]  # Final left side x
        upd_points_right_x = len(points_right_x) * [None]  # Final right side x

        # plt.clf()
        # Plot the lane marker estimate on the hist plot
        for ind, hist_data in reversed(list(enumerate(win_hist_full))):
            plt.subplot(number_of_windows, 1, ind+1)
            plt.xticks([]), plt.yticks([])
            plt.plot(hist_data)
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

                plt.plot(100 * [upd_points_right_x[ind]], np.arange(0,100),linewidth=2.0 )
                plt.plot(100 * [upd_points_left_x[ind]], np.arange(0, 100), linewidth=2.0)

        # plt.show('full_hist.jpg')
        # plt.pause(0.05)

        # for ind, hist_data in enumerate(win_hist_left):
        #     plt.subplot(number_of_windows, 1, ind+1)
        #     plt.xticks([]), plt.yticks([])
        #     plt.plot(hist_data)
        # plt.savefig('left_hist.jpg')

        for ind, hist_data in enumerate(win_hist_right):
            plt.subplot(number_of_windows, 1, ind + 1)
            plt.xticks([]), plt.yticks([])
            plt.plot(hist_data)
        # plt.show()
        # plt.savefig('right_hist.jpg')

        temp_image_copy = bird_view_image.copy()

        # Annotate the points on the image
        for ind, _ in reversed(list(enumerate(upd_points_right_x))):
            cv2.circle(temp_image_copy, (upd_points_right_x[ind], upd_points_right_y[ind]), 6, [0, 255, 0], -1)
            cv2.circle(temp_image_copy, (upd_points_left_x[ind], upd_points_left_y[ind]), 6, [0, 255, 0], -1)
        self.points_im_before = temp_image_copy
        # cv2.imwrite('overlay_on_im.jpg', bird_view_image)

        # Polyfit the points
        # Right lane
        # Get 2nd degree polynomial
        poly_nom = np.polyfit(upd_points_right_y, upd_points_right_x, deg=2)
        lane_polynomial_right = np.poly1d(poly_nom)
        # print(lane_polynomial_right.coeffs)
        # print(lane_polynomial)

        # Compute start and end values of y axis
        start_y = upd_points_right_y[0]
        end_y = upd_points_right_y[-1]
        # Get series of y values
        y_val_right = np.arange(start_y, end_y, 1)

        # Compute x values using the fitted polynomial
        x_val = lane_polynomial_right(y_val_right)
        x_val_right = [int(elem) for elem in x_val]  # Cast to int

        # ======Left lane=================
        # Get 2nd degree polynomial
        poly_nom = np.polyfit(upd_points_left_y, upd_points_left_x, deg=2)
        lane_polynomial_left = np.poly1d(poly_nom)
        # print(lane_polynomial_left.coeffs)

        # Compute start and end values of y axis
        start_y = upd_points_left_y[0]
        end_y = upd_points_left_y[-1]
        # Get series of y values
        y_val_left = np.arange(start_y, end_y, 1)

        # Compute x values using the fitted polynomial
        x_val = lane_polynomial_left(y_val_left)
        x_val_left = [int(elem) for elem in x_val]  # Cast to int

        left_val = [upd_points_left_y, upd_points_left_x]
        right_val = [upd_points_right_y, upd_points_right_x]

        info = self.get_curvature(img.shape, left_val, right_val)

        for ind, _ in enumerate(x_val):
            cv2.circle(temp_image_copy, (x_val_left[ind], y_val_left[ind]), 2, [0, 0, 255], -1)
        for ind, _ in enumerate(x_val):
            cv2.circle(temp_image_copy, (x_val_right[ind], y_val_right[ind]), 2, [0, 0, 255], -1)
        self.points_im_after = temp_image_copy
        return [x_val_right, list(y_val_right)], [x_val_left, list(y_val_left)], info

    @staticmethod
    def get_curvature(image_size, left_val, right_val):

        x_pp = 3.7 / 700
        y_pp = 30 / 720

        y_el = [y_pp * elem for elem in left_val[0]]
        x_el = [x_pp * elem for elem in left_val[1]]
        poly_nom = np.polyfit(y_el, x_el, deg=2)
        left_poly = np.poly1d(poly_nom)

        y_el = [y_pp * elem for elem in right_val[0]]
        x_el = [x_pp * elem for elem in right_val[1]]
        poly_nom = np.polyfit(y_el, x_el, deg=2)
        right_poly = np.poly1d(poly_nom)

        r_coef = right_poly.coeffs
        l_coef = left_poly.coeffs

        left_curve = ((1 + (2 * l_coef[0] * np.max(left_val[0]) * y_pp + l_coef[1]) ** 2) ** 1.5) / np.absolute(
            2 * l_coef[0])
        right_curve = ((1 + (2 * r_coef[0] * np.max(right_val[0]) * y_pp + r_coef[1]) ** 2) ** 1.5) / np.absolute(
            2 * r_coef[0])

        img_height = image_size[0] * y_pp
        img_width = image_size[1] * x_pp

        left_intercept = l_coef[0] * img_height ** 2 + l_coef[1] * img_height + l_coef[2]
        right_intercept = r_coef[0] * img_height ** 2 + r_coef[1] * img_height + r_coef[2]
        center_val = (left_intercept + right_intercept) / 2.0

        vehicle_dev = (center_val - img_width / 2.0)

        # print(left_curve, right_curve, vehicle_dev)
        return [left_curve, right_curve, vehicle_dev]

    def poly_fit_overlay(self, original_im, right_lane_point, left_lane_point, pt):
        x_val_right, y_val_right = right_lane_point
        x_val_left, y_val_left = left_lane_point
        # normal_view_img = pt.get_normal_view(self.bird_view_im)
        norm_right = pt.get_normal_view_points(x_val_right, list(y_val_right))
        norm_left = pt.get_normal_view_points(x_val_left, list(y_val_left))

        blank_img = np.zeros_like(original_im).astype(np.uint8)

        # for ind, _ in enumerate(x_val_left):
        #     cv2.circle(self.normal_point_im, (x_val_left[ind], y_val_left[ind]), 6, [255, 255, 0], -1)
        # for ind, _ in enumerate(x_val_left):
        #     cv2.circle(self.normal_point_im, (x_val_right[ind], y_val_right[ind]), 6, [255, 255, 0], -1)

        self.normal_point_im = blank_img.copy()
        for ind, _ in enumerate(norm_right):
            cv2.circle(self.normal_point_im, (norm_right[ind]), 7, [0, 0, 255], -1)
            cv2.circle(self.normal_point_im, (norm_left[ind]), 7, [0, 0, 255], -1)

        for ind, _ in enumerate(norm_right):
            cv2.circle(blank_img, (norm_right[ind]), 3, [0, 0, 255], -1)
            cv2.circle(blank_img, (norm_left[ind]), 3, [0, 0, 255], -1)


        rev_norm_left = list(reversed(norm_left))
        full_poly = np.array([norm_right + rev_norm_left + [norm_right[0]]], dtype=np.int32)

        cv2.fillPoly(blank_img, full_poly, (0, 255, 0))
        # print(norm_right[0], norm_right[-1])
        # print(norm_left[0], norm_left[-1])
        self.normal_point_im = cv2.addWeighted(original_im, 1, self.normal_point_im, 1, 1)
        result = cv2.addWeighted(original_im, 1, blank_img, 0.3, 0)

        return result

