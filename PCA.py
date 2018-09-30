import cv2
import numpy as np
import h5py
import scipy as scipy
from scipy import sparse
import scipy.sparse.linalg  


def getU(image_data, k):

    print("input shape: s%",image_data.shape)

    # 28*28 images to a 784 vector for each image
    num_pixels = image_data.shape[1] * image_data.shape[2]
    image_data = image_data.reshape(image_data.shape[0], num_pixels)

    # get sigma
    m = len(image_data)
    sigma = (1/m) * np.dot(image_data.T, image_data)

    u, s, vt = scipy.sparse.linalg.svds(sigma, k=k, return_singular_vectors="u")
    print(u.shape)
    return u


labels_training = np.array(h5py.File('labels_training.h5', 'r')['label'])
images_training = np.array(h5py.File('images_training.h5', 'r')['data'])

print(getU(images_training, 300))



