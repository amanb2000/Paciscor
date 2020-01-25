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
LOWEST_CLUSTERS = 20
HIGHEST_CLUSTERS = 20

def read_image(target):
    # FIX DIS
    path = '../../../src/flyers/{}'.format(target)
    path = pt.abspath(pt.join(__file__, path))
    img = cv2.imread(path,0)
    height, width = img.shape
    # Unroll image into intensity
    unroll = img.reshape((height * width, 1))
    return unroll, (height, width)

def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

def bwarea(img):
    row = img.shape[0]
    col = img.shape[1]
    total = 0.0
    for r in range(row-1):
        for c in range(col-1):
            sub_total = img[r:r+2, c:c+2].mean()
            if sub_total == 255:
                total += 1
            elif sub_total == (255.0/3.0):
                total += (7.0/8.0)
            elif sub_total == (255.0/4.0):
                total += 0.25
            elif sub_total == 0:
                total += 0
            else:
                r1c1 = img[r,c]
                r1c2 = img[r,c+1]
                r2c1 = img[r+1,c]
                r2c2 = img[r+1,c+1]
                
                if (((r1c1 == r2c2) & (r1c2 == r2c1)) & (r1c1 != r2c1)):
                    total += 0.75
                else:
                    total += 0.5
    return total
            
def imclearborder(imgBW):

    # Given a black and white image, first find all of its contours
    radius = 2
    imgBWcopy = imgBW.copy()
    image, contours = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    image, contours = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None:
        return imgBWcopy

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy      

def imfill(im_th):
    
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    return im_out

# for i,cluster in enumerate(clusters):
        
#     # Fuzzy C Means
#     new_time = time()
    
#     # error = 0.005
#     # maximum iteration = 1000
#     # cluster = 2,3,6,8
    
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#     rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None,seed=42)

#     new_img = change_color_fuzzycmeans(u,cntr)
    
#     fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
    
#     ret, seg_img = cv2.threshold(fuzzy_img,np.max(fuzzy_img)-1,255,cv2.THRESH_BINARY)
    
#     print('Fuzzy time for cluster',cluster)
#     print(time() - new_time,'seconds')
#     seg_img_1d = seg_img[:,:,1]
    
    
#     bwfim1 = bwareaopen(seg_img_1d, 100)
#     bwfim2 = imclearborder(bwfim1)
#     bwfim3 = imfill(bwfim2)
    
#     print('Bwarea : '+str(bwarea(bwfim3)))
#     print()

#     plt.subplot(1,4,i+2)
#     plt.imshow(bwfim3)
#     name = 'Cluster'+str(cluster)
#     plt.title(name)

#     name = 'segmented'+str(index)+'.png'
#     plt.savefig(name)
#     print()

def run_cluster(targetImg, cluster):
    # Fuzzy C
    trackTime = time()

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(targetImg.T, cluster, 2, error=ALLOWED_ERROR, 
                                                    maxiter=MAX_ITERATIONS, init=None, seed=42)

    return (trackTime, cntr, u, u0, d, jm, p, fpc)

def main():
    # POOL = multiprocessing.Pool(processes=2)

    unrollImg, dimensions = read_image('week_1_page_1.jpg')

    # Format image
    img = np.reshape(unrollImg, (dimensions[0], dimensions[1])).astype(np.uint8)
    shape = np.shape(img)

    # initialize graph
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    # Iterate clusters
    func = partial(run_cluster, unrollImg)
    # result = POOL.map(func, [x for x in range(LOWEST_CLUSTERS, HIGHEST_CLUSTERS+1, 1)])
    
    for i in range(LOWEST_CLUSTERS, HIGHEST_CLUSTERS+1, 1):
        trackTime, cntr, u, u0, d, jm, p, fpc = run_cluster(unrollImg, i)

        # Create Vizualization
        newImg = change_color_fuzzycmeans(u, cntr)
        fuzzyImg = np.reshape(newImg, shape).astype(np.uint8)

        ret, segImg = cv2.threshold(fuzzyImg, np.max(fuzzyImg)-1, 255, cv2.THRESH_BINARY_INV)

        print('Time for {} clusters: {}'.format(i, time() - trackTime))
        segImg1d = segImg[:,:]

        bwfim1 = bwareaopen(segImg1d, 100)
        bwfim2 = imclearborder(bwfim1)
        bwfim3 = imfill(bwfim2)

        print('BWArea: {}'.format(bwarea(bwfim3)))

        plt.subplot(1,4,i+2)
        plt.imshow(bwfim3)
        plt.title('{} Clusters'.format(i))
        plt.savefig()

    print("Done! Best cluster number for flyer is {}".format(2))


#%% Give-err
## Test Execution
if __name__ == "__main__":
    main()