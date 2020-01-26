import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import os.path as pt
import multiprocessing
from time import time

COLORS = [[149, 255, 192], [58, 50, 31], [18, 205, 41], [55, 29, 86], [13, 171, 58], [87, 125, 121], [146, 178, 99], [245, 245, 17], [124, 218, 156],
            [14, 182, 155], [183, 207, 161], [38, 209, 25], [22, 185, 127], [250, 116, 161], [167, 253, 28], [110, 224, 160], [175, 16, 146],
            [101, 62, 167], [75, 247, 89], [248, 66, 241]]

# CONTROLS
ACCURACY = 0.1
MAX_ITERATIONS = 100
MAX_TRIALS = 10
LOWEST_CLUSTERS = 12
HIGHEST_CLUSTERS = 14
TARGET_COMPAT = 270000000000

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

def run_cluster(params):
    # Unpack params
    clusters = params['clusters']
    criteria = params['criteria']
    pixel_values = params['img']
    image = params['src']
    MAX_TRIALS, COLORS = params['CONSTANTS']
    filename = params['fname']

    trackTime = time()

    # Begin calcs

    compat, labels, (centers) = cv2.kmeans(pixel_values, clusters, None, criteria, MAX_TRIALS, cv2.KMEANS_RANDOM_CENTERS)

    # Normalize results
    centers = np.int16(centers)

    labels = labels.flatten()


    # Prep plot data
    final = np.full((image[2][0], image[2][1], 3), 255)
    truthImg = np.full((image[2][0], image[2][1]), -1)

    for pt, val in zip(image[1], labels.flatten()):
        final[pt[0], pt[1]] = COLORS[val]
        truthImg[pt[0], pt[1]] = val

    # Plot
    fig, ax = plt.subplots(1)
    ax.imshow(final)

    # Add rectangles
    boxes = []
    for lbl in range(0, clusters, 1):
        locs = np.where(truthImg==lbl)
        x_min, x_max, y_min, y_max = np.amin(locs[1]), np.amax(locs[1]), np.amin(locs[0]), np.amax(locs[0])
        # print(x_min, x_max, y_min, y_max)
        w = np.cos
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min)
        boxes.append(rect)
    pc = PatchCollection(boxes, facecolor='None', edgecolor="red", linewidths=2)
    ax.add_collection(pc)
        # plt.gca().add_patch(rect[lbl])
        # plt.show()

    # Save
    plt.savefig('{}__K_{}_compat_{}.png'.format(filename, clusters, int(compat)))

    print("Completed computation for {} clusters in {}s".format(clusters, time() - trackTime))

    return clusters, compat, (labels, centers)

POOL = multiprocessing.Pool(processes=5)

tg = 'week_1_page_1.jpg'

image = read_image(tg)

pixel_values = image[1].reshape((-1, 2))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITERATIONS, ACCURACY)

result = POOL.map(run_cluster, [{'fname': tg, 'src': image, 'img': pixel_values, 'criteria': criteria, 'clusters': x, 'CONSTANTS': (MAX_TRIALS, COLORS)} for x in range(LOWEST_CLUSTERS, HIGHEST_CLUSTERS+1, 1)])
POOL.close()
POOL.join()

currentBest = [0, None, None]

for resultSet in result:
    # Compare for best clustering
    if currentBest[1] is None or abs(TARGET_COMPAT - resultSet[1]) < currentBest[1]:
        currentBest = [resultSet[0], abs(TARGET_COMPAT - resultSet[1]), resultSet[2]]

print("It is concluded that the best clustering is {}".format(currentBest[0]))
