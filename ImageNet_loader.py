import numpy as np

def load(batch_name):
	images = []
	imgs = np.load(batch_name)
	num_images = imgs.shape[0]
	for i in range(num_images):
		images.append(imgs[i].transpose(1, 2, 0))
	return images

