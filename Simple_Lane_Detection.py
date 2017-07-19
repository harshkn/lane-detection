from __future__ import division

import cv2
import numpy as np
import matplotlib.pyplot as plt


def showImage(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
def extractROI(image, verticies):
    mask = np.zeros_like(image)
    required_region = (255,) * 3
    cv2.fillPoly(mask, vertices, required_region)
    return mask


cap = cv2.VideoCapture('lane_lines_images/solidYellowLeft.mp4')

fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')
_, sample_image = cap.read()
size = (sample_image.shape[1], sample_image.shape[0])
out = cv2.VideoWriter('Simple_Lane_detector.mp4v', fourcc, 30, size, True)

while cap.isOpened():
    ret, img_color = cap.read()
    if ret is True:

        # Read sample image and convert to hsv format
        # img_color = cv2.imread('lane_lines_images/whiteCarLaneSwitch.jpg')
        img_intermediate = img_color.copy()
        img_final = img_color.copy()
        im_height, im_width, _ = img_color.shape
        # print(img_color.shape)
        img_color = cv2.resize(img_color, (960, 540))
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

        # showImage(img_color)


        # In[38]:

        # Create mask for Yellow and White
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

        lower_white = np.array([0, 0, 230], dtype=np.uint8)
        upper_white = np.array([255, 20, 255], dtype=np.uint8)

        yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(img_hsv, lower_white, upper_white)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        image_masked = cv2.bitwise_or(img_color, img_color, mask=mask)


        # Canny Edge detection
        blur_masked = cv2.GaussianBlur(image_masked, (3,3), 0)
        x_len, y_len, _= img_hsv.shape
        canny_img = cv2.Canny(blur_masked, threshold1=50, threshold2=255, apertureSize=3)
        plt.imshow(canny_img)

        print(img_color.shape)
        # Get Mask to extract region of interest from the canny image

        vertices = np.array([[(0,im_height),(im_width/2, im_height/2), (im_width/2, im_height/2), (im_width,im_height)]], dtype=np.int32)
        # vertices = np.array([[(im_width/2, im_height),(im_width/2, im_height/2), (im_width/2, im_height/2), (im_width,im_height)]], dtype=np.int32)
        roi_mask = extractROI(canny_img, vertices)
        roi_image = cv2.bitwise_and(canny_img, roi_mask)



        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(roi_image,3,np.pi/180,20,minLineLength)

        img_intermediate = img_color.copy()
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                if(x2 -x1) != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    if np.fabs(slope) < 0.2:
                        continue
                    elif slope < 0 : # left lane, green is left
                        cv2.line(img_intermediate,(x1,y1),(x2,y2),(0,255,0),2)
                        left_x.extend([x1, x2])
                        left_y.extend([y1, y2])
                    elif slope > 0:  # right lane, red is right, it rhymes ;)
                        right_x.extend([x1, x2])
                        right_y.extend([y1, y2])
                        cv2.line(img_intermediate,(x1,y1),(x2,y2),(0,0,255),2)

        # showImage(img_intermediate)


        # In[77]:

        # let's plot the right x and y coordinates as a scatter plot.
        # plt.scatter(right_x, right_y)
        # plt.show()
        # plt.scatter(left_x, left_y)
        # plt.show()
        # To obtain a single line , we can apply linear regression to fit a line to the points.
        # We start by fitting a polynomial of degree one and we get a polynomial for right lane
        deg_1_poly = np.polyfit(right_y, right_x, deg=1)
        right_lane_polynomial = np.poly1d(deg_1_poly)
        print('Right lane polynomial :' + str(right_lane_polynomial))

        deg_1_poly = np.polyfit(left_y, left_x, deg=1)
        left_lane_polynomial = np.poly1d(deg_1_poly)
        print('Left lane polynomial :' + str(left_lane_polynomial))

        # Get a single line from the polynomial. Since the line cannot extend from bottom to the top of the image,
        # let us limit the line to region of interest
        min_y = int(img_final.shape[0]/1.7)
        max_y = int(img_final.shape[0] )
        right_x_start = int(right_lane_polynomial(max_y))
        right_x_end = int(right_lane_polynomial(min_y))

        left_x_start = int(left_lane_polynomial(max_y))
        left_x_end = int(left_lane_polynomial(min_y))



        img_final = img_color.copy()
        # cv2.line(img_final, (right_x_start, max_y), (right_x_end, min_y), (0, 255,0), 4)
        # cv2.line(img_final, (left_x_start, max_y), (left_x_end, min_y), (0, 0,255), 4)
        # showImage(img_final)

        just_line = np.zeros(img_final.shape, dtype=np.uint8);
        cv2.line(just_line, (right_x_start, max_y), (right_x_end, min_y), (0, 0,255), 4)
        cv2.line(just_line, (left_x_start, max_y), (left_x_end, min_y), (0, 255,0), 4)

        img_final = cv2.addWeighted(img_final, 0.8, just_line, 1, 0);
        # showImage(img_final)
        cv2.imshow('Main Window', img_final)
        cv2.waitKey(100)

        out.write(img_final)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()