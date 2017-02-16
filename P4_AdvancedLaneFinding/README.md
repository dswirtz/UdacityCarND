#Udacity Self-Driving Car Nanodegree - Adv. Lane Finding

#####Douglas Wirtz
#####February 13th, 2017

##Background

Computer vision is the science of gaining a high-level understanding of the world through digital images and videos. It is one of the many important approaches in successfully engineering a self-driving car. The images and videos used for this project were obtained using a front-facing camera on a moving car. The goal of this project is to use various computer vision techniques in a software pipeline to identify the lane boundaries on an image of a road. This is followed by an application of the pipeline on a series of images (i.e. video) to assess the robustness of the software.

##Approach

I decided on a basic step-wise approach for this project, completing each task by building on the previous ones. Outlined below are the series of steps I took to complete this project. To view a more in-depth detail of the code used, the steps are clearly numbered in the [IPython notebook](). 

**Advanced Lane Finding Project**

1. Import dependencies and functions
2. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
3. Apply a distortion correction to raw images
4. Use color transforms, gradients, etc., to create a thresholded binary image
5. Apply a perspective transform to rectify binary image ("birds-eye view")
6. Detect lane pixels and fit to find the lane boundary
7. Determine the curvature of the lane and vehicle position with respect to center
8. Warp the detected lane boundaries back onto the original image
9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
10. Run pipeline on a video

###1. Import dependencies and functions

Here I imported all the dependencies and funtions I used to complete this project. The idea here was to keep the IPython notebook organized by keeping the individual steps clean. This also proved to be beneficial because it allowed me to build the `pipeline()` funtion piece by piece. During experimentation, I built the function in the step it was used in, and I made changes as I saw fit. Once it was finalized, I moved it the dependencies and functions code block.

###2. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images

The absolute first step in this project must be camera calibration. This will allow you to correct for image distortion and accurately process each image to be representative of real world distances. 

In order to do this, I filled a list of object points, `objpoints = []`, which are 3D points in a real world space, and a list of image points, `imgpoints = []`, which are the corresponding 2D points in an image. I filled the lists using a combination of the set of 9x6-corner chessboard calibration images in the `/camera_cal` folder and the `cv2.findChessboardCorners()` function. If the corners were found in the image, the real world object points and corresponding image points were appended to their respective lists. While looping through the images, finding the chessboard corners, I used the `cv2.drawChessboardCorners()` function to draw the corners on the chessboard images. Here is an example:

![image](img)

Finally, I used the `cv2.calibrateCamera()` function in conjunction with `objpoints` and `imgpoints` to compute the calibration matrix, `mtx`, and the distortion coefficients, `dist`.

###3. Apply a distortion correction to raw images

Along with `mtx` and `dist` from the previous section, I used `cv2.undistort()` to undistort images. Once an image is corrected for distortion, it can then proceed to the meat of image processing. Below is an example of an undistorted calibration image as well as an example of a test image from the `/test_images` folder.

![image](img)

![image](img)

###4. Use color transforms, gradients, etc., to create a thresholded binary image

###5. Apply a perspective transform to rectify binary image ("birds-eye view")

###6. Detect lane pixels and fit to find the lane boundary

###7. Determine the curvature of the lane and vehicle position with respect to center

###8. Warp the detected lane boundaries back onto the original image

###9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

###10. Run pipeline on a video

***
