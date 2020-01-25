import cv2 
import pytesseract
import numpy as np
from pytesseract import Output

from preprocessing import *

def get_data(path: str, coords: tuple, conf=r'--oem 3 --psm 3 -l eng', debug=False):
    raw_img = cv2.imread(path)

    crop_img = raw_img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]

    red_img = crop_img

    red_img = get_red(red_img)

    crop_img = get_grayscale(crop_img)
    crop_img = thresholding(crop_img)

    if debug:
        cv2.imshow("Red image", red_img)
        cv2.waitKey(0)


    s = pytesseract.image_to_string(crop_img, config=conf)
    d = pytesseract.image_to_data(crop_img, output_type=Output.DICT, config=conf)

    return(crop_img, d)

# def get_red_text(path: str, coords: tuple, conf=r'--oem 3 --psm 3 -l eng', debug=False):


coords = ((917, 300), (1355, 868))
img, data = get_data('py-testing/week_10_page_1_cropped_png.png', coords, debug=True)

print('Data lengths: {} {} {}'.format(len(data['level']),len(data['conf']),len(data['text']) ))

line_1 = []
strOut = ''

# Drawing boxes around recognized words. 
for i in range(0, len(data['level'])):
    if int(data['conf'][i]) > 60:
        line_1 += [data['text'][i]]
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(line_1)

# print(data['conf'])

cv2.imshow('img', img)
cv2.waitKey(0)
