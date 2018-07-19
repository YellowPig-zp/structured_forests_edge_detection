import os 
import cv2 as cv

path = "./BSR/BSDS500/data/images/train/"

def load():
	images = []
	for filename in os.listdir(path):
		if filename == "Thumbs.db":
			continue
		filename = path + filename
		img = cv.imread(filename)
		# img is fed into cv method, which deals with BGR images
		# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		images.append(img)
	return images, None

