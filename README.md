#### Self Driving Car - Simple and Advanced Lane Detection.

This project was part of Udacity Nanodegree. I have not taken the course but trying to complete all the project assignments in the course. Lane detection (Simple and Advanced) are part of assignment in Term 1

##### Advanced Lane Detection

These are the goals of this project taken from the Udacity assignment page.  
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.  
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.  

###### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Initial step is to compute the camera calibration matrix to undistort the images. This is done by using a set of chess board pattern images in which chess board pattern is on a plane surface. Atleast 10 chessboard pattern images are required for camera calibration matrix computation.  

There are various kinds of distortion that could occur in the images taken and the ones we consider correcting in this problem are  radial and tangential distortion. Radial distortion results in straight lines appearing curved/bulged out. The bulging increases with the distance from the center. Tangential distortion occurs due to misalignment of lens wrt projection plane(or imaging plane). This results in some areas seeming nearer than other. 

Camera intrinsic and extrinsic parameters are also required to perform camera calibration. Intrinsic parameters are as the name suggests, intrinsic to the camera. That is the focal length, camera principle point and skew. Extrinsic parameters include the rotation and translation matrices. 


OpenCV api for camera calibration returns distortion coefficients along with camera calibration matrix. 

 
<img src="images/orig.jpg" width="400"/><img src="images/undist.jpg" width="400"/>






##### Simple Lane Detection

- Hello 1
- Hello 2

