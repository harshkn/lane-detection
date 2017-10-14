from __future__ import division
import cv2 as cv2
import glob
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import cPickle
from sklearn.metrics import classification_report

class VehicleDetectorTrainer:

    def __init__(self):
        self.hello = 1
        self.path_cars =  "/Users/harshkn/happyhour/SelfDrivingCar/data/vehicles/*/*.png"
        self.path_no_cars = "/Users/harshkn/happyhour/SelfDrivingCar/data/non-vehicles/*/*.png"
        self.X = []
        self.y = []
        self.clf = None

        try:
            # save the classifier
            with open('clf_car_not_car.pkl', 'rb') as fid:
                self.clf = cPickle.load(fid)
        except:
            print('Error..')
            # self.load_dataset()
            # self.train_test()

    def load_dataset(self):
        cars = glob.glob(self.path_cars)
        non_cars = glob.glob(self.path_no_cars)
        print(len(cars))
        print(len(non_cars))

        for file_name in cars:
            img = cv2.imread(filename=file_name)
            assert img.shape == (64, 64, 3)
            feat = self.get_features(img)
            self.X.append(feat)
            self.y.append(1)

        for file_name in non_cars:
            img = cv2.imread(filename=file_name)
            assert img.shape == (64, 64, 3)
            feat = self.get_features(img)
            self.X.append(feat)
            self.y.append(0)

    def train_test(self):


        # X = nnp.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        clf = svm.SVC(kernel='linear', C=1.0, probability=True)
        clf.fit(X_train, y_train)

        # save the classifier
        with open('clf_car_not_car.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)

        y_pred = []
        for sample in X_test:
            t_sample = np.array([sample])
            pred = clf.predict(t_sample)
            y_pred.append(pred)
        print('Accuracy is %.2f' % accuracy_score(y_test, y_pred))

    @staticmethod
    def get_features(image):
        fd = []
        ft = hog(image[:, :, 0])
        ft = ft.flatten()
        fd.append(ft)
        ft = hog(image[:, :, 1])
        ft = ft.flatten()
        fd.append(ft)
        ft = hog(image[:, :, 2])
        ft = ft.flatten()
        fd.append(ft)
        fd = np.concatenate(fd).ravel()
        return fd

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]

        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        else:
            y_start_stop[0] = int(img.shape[0] * y_start_stop[0] / 100.)

        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]
        else:
            y_start_stop[1] = int(img.shape[0] * y_start_stop[1] / 100.)

        print("Y start and stop: {}".format(y_start_stop))

        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def search_windows(self, img, windows):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.get_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            features = np.array(features).reshape(1, -1)
            # test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = self.clf.predict(features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        # imcopy = img.copy()
        for bbox in bboxes:
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)

if __name__ == '__main__':
    vd = VehicleDetectorTrainer()
    print(vd.clf)

    im_path = "/Users/harshkn/happyhour/SelfDrivingCar/test_images/test6.jpg"
    test_img = cv2.imread(im_path)

    all_windows = vd.slide_window(test_img)
    boxes = vd.search_windows(test_img, all_windows)
    vd.draw_boxes(test_img, boxes)
    print(boxes)

    cv2.imshow("Win", test_img)
    cv2.waitKey(0)


