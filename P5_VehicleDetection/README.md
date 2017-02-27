#Udacity Self-Driving Car Nanodegree - Vehicle Detection

#####Douglas Wirtz
#####February 27th, 2017

##Background

Computer vision is the science of gaining a high-level understanding of the world through digital images and videos. It can be used along with a well-trained classifier to develop a powerful approach to engineer a self-driving car. The test images and videos used for this project were obtained using a front-facing camera on a moving car. The [car](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and [non-car](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip) datasets used for training the classifier were provided by Udacity. The goal of this project is to use the car and non-car datasets to train a classifier to detect vehicles on a road. Then using various computer vision techniques and the trained model, I built a software pipeline to detect and track the vehicles in both single images and videos.

##Approach

Similar to P4_AdvancedLaneLines, I approached this project by breaking the image processing video pipeline into its components. This allowed me to explore/visualize the data, experiment with ideas, and tune a pipeline one step at a time. Outlined below are the series of steps I took to complete this project. To view a more in-depth detail of the code used, the steps are clearly numbered in **P5_VehicleDetection.ipynb**. 

**Vehicle Detection Project**

1. Import dependencies and functions
2. Read in dataset images and process for visualization
3. Explore image features
4. Train a classifier
5. Apply classifier to predict vehicle detection in test images
6. Run classifier pipeline on project video

###1. Import dependencies and functions

Here I imported all the dependencies and funtions I used to complete this project. The idea here was to keep the IPython notebook organized by keeping the individual steps clean. This also proved to be beneficial because it allowed me to build the `pipeline()` funtion piece by piece. During experimentation, I built the function in the step it was used in, and I made changes as I saw fit. Once it was finalized, I moved it to the dependencies and functions code block.

###2. Read in dataset images and process for visualization

To read in the datasets, I used the `glob` module to generate a list of file paths. Then with the defined function `read_images()`, I read in each image using `mpimg.imread()`. After the images were read in, I checked the number in each dataset and found that the number of cars and non-cars were relatively similar, so I didn't take any action to even them out. I then doubled the size of the datasets with my function `flip_append()`, which essentially loops through every image, flips it, and appends the original along with the flipped into a new list. The final action I took with the classification images was to fix the scaling of the images. Using `mpimg.imread()` reads in .PNG images scaled 0 to 1. I fixed this with my `fix_scale()` function which loops through all of the images in a list and multiplies each image by 255. Here is an example of a car and non-car image with its flipped counterpart.

![img](image)

###3. Explore image features

In this step, I used the original car and non-car examples from above and explored the different features. First up was exploring the color channel features. Using the function `clolor_hist()` with `vis_hist=True`, it took in an RGB images and created histograms of each color channel. In addition, the function returns a list of histogram features which can be visualize with a plot. With this visualization alone, you can tell images of cars have much more color differential when compared to an image of a non-car.

![img](image)

Next up I looked at spatially binned features of an RGB version of the car and non-car example. 
###4. Train a classifier

###5. Apply classifier to predict vehicle detection in test images

###6. Run classifier pipeline on project video

###Discussion

On the first couple of implementations of my pipeline, I remember two distinct problems I faced. The first of which was an issue occuring when the car enters/leaves shaded areas. The lane detection worked fairly well up to those points, and then, in my opinion, failed miserably. The detection would become very wobbly and flicker in and out. After some research and adjustment of some thresholds, I implemented a yellow/white color filter to the binary output. This allows the pipeline adjust better to those sudden changes in lighting. 

The second problem I remember having was the passing car on the right. As the car was passing, my early implementation picked up the car as a lane line, essentially making my lane twice as big. To fix this issue, I implemented an area-of-interest mask to my binary output that zeroed all pixels outside of the area. This issue was immediately resolved in my next implementation. I did make other changes since this addition, so it's possible I might have been able to solve this problem using other means.

My final implementation works fairly well, but there is always a way to make it more robust. In a future implementation, I would like to work on making the right lane (or dashed lane lines in general) perform better. Another improvement I would like to make is make the curvature of each lane line more consistent with its counterpart.

This current implementation/pipeline will likely fail on dirt roads, or on roads with harsh conditions (i.e. weather or anything that would make it difficult to see the lane lines). Another thing to consider is that this pipeline was run on a video of a well-kept highway. Over time, road integrity will diminish. If the road integrity diminishes to a point where we couldn't see the line, this implementation will likely fail. So to add to the future goals of this pipeline, I would implement an algorithm, maybe even a deep neural network, to predict what the lane line would look like given a very little amount of information.

***
