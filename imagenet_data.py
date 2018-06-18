import settings
import os
import numpy as np

# number of images to be picked per imagenet batch
NUM_IMAGES_PICKED = 25

ORIGINALS_PATH = settings.ORIGINAL_IMAGE_PATH
EDGEMAPS_PATH = settings.EDGEMAP_PATH

def generate_picked_dataset():
	"""
	Randomly pick NUM_IMAGES_PICKED images and corresponding 
	edgemaps per imagenet batch and save them to ORIGINALS_PATH 
	and EDGEMAPS_PATH
	"""
	for batch_name in os.listdir(ORIGINALS_PATH):
		if batch_name in os.listdir(settings.PICKED_ORIGINALS_PATH):
			print("{} done already!".format(batch_name))
			continue
	
		original_imgs = np.load(ORIGINALS_PATH + batch_name)
		edgemaps = np.load(EDGEMAPS_PATH + batch_name)

		assert original_imgs.shape[0] == edgemaps.shape[0]

		num_imgs = original_imgs.shape[0]
		picked_indices = np.random.choice(num_imgs, NUM_IMAGES_PICKED, replace=False)
		original_imgs_picked = original_imgs[picked_indices].transpose(0, 2, 3, 1)
		edgemaps_picked = edgemaps[picked_indices]

		np.save(settings.PICKED_ORIGINALS_PATH+batch_name, original_imgs_picked)
		np.save(settings.PICKED_EDGEMAPS_PATH+batch_name, edgemaps_picked)

		print("{} done!".format(batch_name))

def generate_single_files_dataset():
	"""
	Bundle all the images and edgemaps into two separate
	.npy files and save them to settings.DATASET_PATH
	"""
	original_imgs_picked = None
	edgemaps_picked = None

	for batch_name in os.listdir(settings.PICKED_ORIGINALS_PATH):

		original_imgs = np.load(settings.PICKED_ORIGINALS_PATH+batch_name)
		edgemaps = np.load(settings.PICKED_EDGEMAPS_PATH+batch_name)

		if original_imgs_picked is None and edgemaps is None:
			original_imgs_picked = original_imgs
			edgemaps_picked = edgemaps
		else:
			original_imgs_picked = np.concatenate((original_imgs_picked, original_imgs), axis=0)
			edgemaps_picked = np.concatenate((edgemaps_picked, edgemaps), axis=0)

	np.save(settings.DATASET_PATH+"original_images.npy", original_imgs_picked)
	np.save(settings.DATASET_PATH+"edgemaps.npy", edgemaps_picked)

def get_picked_imagenet_dataset():
	"""
	Returns a tuple (ORIGINAL_IMAGES, EDGEMAPS)
	"""


