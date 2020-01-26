import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
            if invImg[i,j] > 100:
                mappedImg.append([i, j])
    return invImg, np.array(mappedImg), (height, width)

image = read_image('week_1_page_1.jpg')
# Invert

pixel_values = image[1].reshape((-1, 2))
pixel_values = np.float32(pixel_values)
# print(pixel_values.shape)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 18

_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
# print(centers)

centers = np.int16(centers)

print(centers)

# flatten the labels array
# print(labels)

colors = [[149, 255, 192], [58, 50, 31], [18, 205, 41], [55, 29, 86], [13, 171, 58], [87, 125, 121], [146, 178, 99], [245, 245, 17], [124, 218, 156],
            [14, 182, 155], [183, 207, 161], [38, 209, 25], [22, 185, 127], [250, 116, 161], [167, 253, 28], [110, 224, 160], [175, 16, 146],
            [101, 62, 167], [75, 247, 89], [248, 66, 241]]

labels = labels.flatten()

pts = centers[labels.flatten()]

final = np.full((image[2][0], image[2][1], 3), 255)

for pt, val in zip(image[1], labels.flatten()):
    final[pt[0], pt[1]] = colors[val]

print(final)
# print(labels)

# segmented_image = centers[labels.flatten()]

# segmented_image = segmented_image.reshape(image.shape)
# # show the image
plt.imshow(final)
plt.savefig('testing.png')

# for pts in centers:
#     plt.plot(pts[0], pts[1], 'ro')
# plt.savefig('testing.png')
