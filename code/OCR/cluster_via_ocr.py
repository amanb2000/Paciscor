import cv2 
import pytesseract
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from pytesseract import Output
from tqdm import tqdm

from preprocessing import *
from ocr_library import *




def get_map(path, debug = False, step_size = 50, radius = 100):
    print('\n\nStarting blurred map OCR process... \n\n')

    img, dict_in = get_data(path, conf=r'--oem 1 --psm 11', debug=debug) # No coordinates given because we are analyzing the entire picture.
    
    df = get_centers(dict_in[1]) # Getting the central coordinates of every single word (along with the confidene and dimensions of the bounding box).

    im_out = get_blurred_map(df, img, step_size, radius)
    
    return im_out

# def goodness_of_centroids(heat_map, points): # TODO: Get this function done
    # heat map is an OpenCV image and points is 


if __name__ == '__main__':
    a = get_map('py-testing/week_24_page_1.png', debug = True)

    cv2.imshow('Centroid Test', a)
    cv2.waitKey(0)