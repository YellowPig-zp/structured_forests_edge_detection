import settings
import os
import numpy as np

# number of images to be picked per imagenet batch
NUM_IMAGES_PICKED = 25

ORIGINALS_PATH = settings.ORIGINAL_IMAGE_PATH
EDGEMAPS_PATH = settings.EDGEMAP_PATH
def generate_dataset():
	for batch_name in os.listdir(ORIGINALS_PATH):
		orignal_imgs = np.load(ORIGINALS_PATH + batch_name)
		edgemaps = np.load(EDGEMAPS_PATH + batch_name)

		assert orignal_imgs.shape[0] == edgemaps.shape[0]

		num_imgs = orignal_imgs.shape[0]
		picked_indices = np.random.choice(num_imgs, NUM_IMAGES_PICKED, replace=False)
		orignal_imgs_picked = orignal_imgs[picked_indices].transpose(0, 2, 3, 1)
		edgemaps_picked = np.array([edgemaps[picked_indices]])

		np.save(settings.PICKED_ORIGINALS_PATH+batch_name, orignal_imgs_picked)
		np.save(settings.PICKED_EDGEMAPS_PATH+batch_name, edgemaps_picked)
