import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as pt

def read_image(target):
    # FIX DIS
    path = '../../../src/flyers/{}'.format(target)
    path = pt.abspath(pt.join(__file__, path))
    img = cv2.imread(path, 0)
    height, width = img.shape
    return img, (height, width)

image = read_image('week_1_page_1.jpg')
# Invert
invImg = cv2.bitwise_not(image[0])
image = cv2.cvtColor(invImg, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4

_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
print(centers)

centers = np.uint8(centers)

print(centers)

# flatten the labels array
print(labels)

labels = labels.flatten()

print(labels)

# segmented_image = centers[labels.flatten()]

# segmented_image = segmented_image.reshape(image.shape)
# # show the image
# plt.imshow(segmented_image)
# plt.show()

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
for cluster in range(0,3,1):
    masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
# show the image
plt.imshow(masked_image)
plt.show()