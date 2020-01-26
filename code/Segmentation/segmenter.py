#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os.path as pt
import cv2
import numpy as np
import multiprocessing
from functools import partial
from time import time

# CONTROLS
ALLOWED_ERROR = 0.005
MAX_ITERATIONS = 2
LOWEST_CLUSTERS = 10
HIGHEST_CLUSTERS = 20

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
    mappedImg = np.empty((height * width, 3))
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            mappedImg[i*width + j] = [i, j, invImg[i,j]]
    return mappedImg, (height, width)

def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

def run_cluster(targetImg, cluster):
    # Fuzzy C
    trackTime = time()

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(targetImg.T, cluster, 2, error=ALLOWED_ERROR, 
                                                    maxiter=MAX_ITERATIONS, init=None, seed=42)

    return cluster, trackTime, cntr, u, fpc

def main():
    POOL = multiprocessing.Pool(processes=4)

    srcImg, dimensions = read_image('week_1_page_1.jpg')

    # initialize graph
    # plt.figure(figsize=(20,20))
    # img = srcImg.reshape(dimensions[0], dimensions[1])
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.savefig('blacked')

    # Iterate clusters
    func = partial(run_cluster, srcImg)
    result = POOL.map(func, [x for x in range(LOWEST_CLUSTERS, HIGHEST_CLUSTERS+1, 1)])
    POOL.close()
    POOL.join()

    for resultSet in result:
        cluster, trackTime, cntr, u, fpc = resultSet

        # Create Vizualization
        # newImg = change_color_fuzzycmeans(u, cntr)
        # fuzzyImg = np.reshape(newImg, shape).astype(np.uint8)

        # ret, segImg = cv2.threshold(fuzzyImg, np.max(fuzzyImg)-1, 255, cv2.THRESH_BINARY_INV)

        print('Time for {} clusters: {}'.format(cluster, time() - trackTime))
        # segImg1d = segImg[:,:]

        # bwfim1 = bwareaopen(segImg1d, 100)
        # bwfim2 = imclearborder(bwfim1)
        # bwfim3 = imfill(bwfim2)

        # print('BWArea: {}'.format(bwarea(bwfim3)))
        # for i in newImg:
        #     plt.plot(i[0], i[1], '.', color='b')

        centers = np.uint8(cntr)

        print(centers)

        for pt in cntr:
            plt.plot(pt[0], pt[1], pt[2], 'rs', projection='3d')

        plt.title('{} Clusters, FPC: {}'.format(cluster, fpc))
        plt.savefig('regions_{}_clusters.png'.format(cluster))

    print("Done! Best cluster number for flyer is {}".format(2))


#%% Give-err
## Test Execution
if __name__ == "__main__":
    main()