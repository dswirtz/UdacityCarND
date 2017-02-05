# This file contains the utilities used for image processing
# and data generation
import cv2
import numpy as np
from scipy.misc import imresize
import matplotlib.image as mpimg

def process_img(image):
	# crop and resize the images
	img_resize = []
	for img in image:
		img = img[60:140, :, :]
		img_resize.append(imresize(img, (64, 64, 3)))
	# convert images to grayscale
	img_gray = np.mean(img_resize, axis=3, keepdims=True)
	# normalize the images
	img_processed = (img_gray - 128.0) / 128.0
	return img_processed
	
def mirror_data(image, angle):
	# randomly mirror image and angle
	r = np.random.choice(2)
	if r == 1:
		image = np.fliplr(image)
		angle = -angle
	return image, angle
	
	# based on a function by Vivek Yadav
def augment_brightness(image):
	# apply random brightness augmentation
	image_aug = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	r = .25+np.random.uniform()
	image_aug[:,:,2] = image_aug[:,:,2]*r
	image_aug = cv2.cvtColor(image_aug,cv2.COLOR_HSV2RGB)
	return image_aug
	
	# based on a function by Vivek Yadav 
def trans_image(image, angle, trans_range):
	# apply random brightness augmentation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    angle_tr = angle + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    return image_tr, angle_tr

def angle_threshold(image, angle):
	# randomly drop 0-angles to decrease the pobability of predicting zero
	r = np.random.choice(2)
	if abs(angle) == 0.0 and r == 1:
		return 9999.0, 9999.0
	else:
		return image, angle
	
	# batch generation
def generate_batch(data, batch_size):
	img_num = len(data)
	while 1:
		indx = np.random.choice(img_num, batch_size)
		batch_data = data[indx]
		batch_images = []
		batch_angles = []
		for line in batch_data:	
			image = mpimg.imread(line[0], 'rb')
			angle = float(line[1])
			image, angle = mirror_data(image, angle)
			image, angle = trans_image(image, angle, 100)
			image = augment_brightness(image)
			image, angle = angle_threshold(image, angle)
			if angle != 9999.0:
				batch_images.append(image)
				batch_angles.append(angle)
		
		x = process_img(np.array(batch_images).astype('float'))
		y = np.array(batch_angles)
	
		yield (x, y)