import numpy as np
from scipy.misc import imresize
import cv2

# Convert state to image
def obs2img(s, image_name):
	cv2.imwrite(image_name, s)

# Downsample Montezuma Revenge state
def downsample(s, save=None):
	resized = imresize(s.astype(np.float64), (48, 48, 3))
	grayscale = np.mean(resized, axis=2).astype(np.uint8)
	if save != None:
		obs2img(grayscale, save)
	return np.resize(grayscale, (1, 48, 48))