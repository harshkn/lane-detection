from __future__ import division
import numpy as np
import cv2
import glob
import os


class ImageWarper:

    def __init__(self):
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

        self.M = cv2.getPerspectiveTransform(src=src, dst=dst)
        self.M_inv = cv2.getPerspectiveTransform(src=dst, dst=src)

    def get_bird_view(self, img):
        return cv2.warpPerspective(img, self.M, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def get_normal_view(self, img):
        return cv2.warpPerspective(img, self.M_inv, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def get_normal_view_points(self, x_val, y_val):
        temp = len(x_val) * [1]

        ret_list = []
        new_points = [(p, y_val[ind], temp[ind]) for ind, p in enumerate(x_val)]
        # print(new_points)
        for p in new_points:
            p = np.array(p)
            new_point = np.dot(self.M_inv, p)
            new_point = new_point / new_point[2]
            ret_list.append((int(new_point[0]), int(new_point[1])))
        return ret_list



if __name__ == "__main__":
    cp = os.getcwd()
    path = 'test_images'
    images = glob.glob(os.path.join(cp, path) + '/*.jpg')
    image = cv2.imread(images[1])

    imw = ImageWarper()
    warped_img = imw.get_bird_view(image)
    print(np.max(warped_img))
    cv2.imshow('win_1', image)
    cv2.imshow('win_2', warped_img)
    cv2.waitKey(0)