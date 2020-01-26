import cv2 
import pytesseract
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from pytesseract import Output
from tqdm import tqdm

from preprocessing import *
from ocr_library import *









if __name__ == '__main__':
    toggle = False

    if toggle:
        a = get_map('py-testing/week_24_page_1.png', debug = True)

        with open('heat_map.pickle', 'wb') as f:
            pickle.dump(a, f)

        cv2.imshow('Centroid Test', a)
        cv2.waitKey(0)
    else:
        lob = ''
        with open('heat_map.pickle', 'rb') as f:
            lob = pickle.load(f)
            lob = cv2.cvtColor(lob,cv2.COLOR_GRAY2RGB)

        coords = ((409, 2338), (875, 3024))

        avg = get_heat_density(lob, coords)

        print('AVERAGE HEAT INTENSITY: {}'.format(avg))

        lab = cv2.rectangle(lob, (coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]), (0, 0, 255), 2)

        cv2.imshow('Centroid Test', lab)
        cv2.waitKey(0)