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
    # Unroll image into intensity
    # unrollImg = img.reshape((height * width), 3)
    # Invert
    invImg = cv2.bitwise_not(img)
    mappedImg = []
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            if invImg[i,j] > 200:
                mappedImg.append([i, j])
    return np.array(mappedImg), (height, width)

image = read_image('week_1_page_1.jpg')
# Invert

pixel_values = image[0].reshape((-1, 2))
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 15

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

for pts in centers:
    plt.plot(pts[0], pts[1], 'ro')
plt.savefig('testing.png')