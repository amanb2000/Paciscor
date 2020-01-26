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
from ocr_script import *



if __name__ == '__main__':
    print('\n\nStarting blurred map acquisition process... \n\n')
    dict_in = get_data('py-testing/week_24_page_1.png')[1]
    
    df = get_centers(dict_in)

    img = dict_in['image']

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    cnt = 0

    im_out = get_blurred_map(df, img, 100, 200)

    



    cv2.imshow('Test Rectangles', im_out)
    cv2.waitKey(0)



