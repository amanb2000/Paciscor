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

def get_data(path: str, coords=False, conf=r'--oem 1 --psm 11', debug=False):
    print('Beginning to get data...')
    raw_img = cv2.imread(path)

    if(coords):
        crop_img = raw_img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
    else:
        crop_img = raw_img

    red_img = crop_img

    red_img = get_red(red_img)

    # red_data = get_red_data(red_img)

    crop_img = get_grayscale(crop_img)
    crop_img = thresholding(crop_img)

    if debug:
        cv2.imshow("Red image", red_img)
        cv2.waitKey(0)


    print('Getting image data for RGB and red filtered... ')
    s = pytesseract.image_to_string(crop_img, config=conf)
    d = pytesseract.image_to_data(crop_img, output_type=Output.DICT, config=conf)
    r = pytesseract.image_to_data(red_img, output_type=Output.DICT, config=conf)
    rs = pytesseract.image_to_string(red_img, config=conf)
    print('Done getting data!\n')

    ret_image = raw_img

    if coords:
        ret_image = raw_img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]

    ret_val = (
        {
            'text': rs,
            'type': 1,
            'data': r,
            # 'image': ret_image
        },
        {
            'text': s,
            'type': 2,
            'data': d,
            # 'image': ret_image
        }
    )

    return(ret_val)

def visualize_results(dict_in):
    img = dict_in['image']
    data = dict_in['data']

    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    bounds = (0, 33, 0, 100, 0)
    # Proper green
    # Red
    # Blue
    # Purple
    # Yellow
    # Light green

    for i in range(0, len(data['level'])):
        if int(data['conf'][i]) > 0:
            color = (200, 255, 200)
            if data['height'][i] <= bounds[0]:
                color = (0, 255, 0)
            elif data['height'][i] <= bounds[1]:
                color = (0, 0, 255)
            elif data['height'][i] <= bounds[2]:
                color = (255, 0, 0)
            elif data['height'][i] <= bounds[3]:
                color = (255, 0, 255)
            elif data['height'][i] <= bounds[4]:
                color = (0, 255, 255)
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # print(dict_in['text'])

    # print(data['conf'])

    name = ''
    if dict_in['type'] == 1:
        name = 'RED IMAGE'
    else:
        name = 'NORMAL IMAGE'

    cv2.imshow(name, img)
    cv2.waitKey(0)

def height_histogram(dict_in):
    df = pd.DataFrame(dict_in['data'])

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    n_bins = 50
    max_height = 50

    trimmed_df = df[df['height'] < max_height]
    # trimmed_df = trimmed_df[trimmed_df['height'] < 625]

    df = trimmed_df

    axs[0].hist(df['height'].to_numpy(), bins=n_bins)

    axs[1].hist(df['height'].to_numpy(), bins = n_bins)

    plt.show()

def get_centers(dict_in, min_conf = 60):

    print('Getting centers...')

    df = pd.DataFrame(dict_in['data'])
    df = df.astype({'conf':'float32'})
    df = df[df['conf'] > 60]

    df['center-x'] = df['left'] + df['width']/2
    df['center-y'] = df['top'] + df['height']/2
    df['area'] = df['width'] * df['height']

    return df[['center-x', 'center-y', 'area', 'conf']]

def get_blurred_map(df, img, step_size, radius):

    print('Getting blurred map...')

    im_out = img
    for i in tqdm(range(0, img.shape[0], step_size)):
        # print(i)
        for j in range(0, img.shape[1], step_size):
            # Getting the sum of the confidences where the word is within the given radius:

            cnt = 0
            for index, row in df.iterrows():
                if abs(row['center-x']-i) < radius and abs(row['center-y']-j) < radius:
                    cnt += 1

            cnt *= 20
            avg_conf = min(cnt, 255)


            # im_out = cv2.rectangle(im_out, (i, j), (i+step_size, j + step_size), (0, 0, avg_conf), int(step_size/3))

            for k in range(max(0, int(i-step_size/2)), min(int(i+step_size/2), img.shape[0]) ):
                for l in range(max(0, int(j-step_size/2)), min(int(j+step_size/2), img.shape[1]) ):
                    im_out[k][l] = (0, 0, avg_conf)

    im_out = get_grayscale(im_out)

    img = get_grayscale(img)
    print('Done getting blurred map.')
    print('Applying blurred map to pre-existing image for reference...')

    for i in tqdm(range(0, img.shape[0])):
        for j in range(0, img.shape[1]):
            im_out[i][j] *= min(min(255, int(im_out[i][j]*2)), int(img[i][j]))
    
    print('Done!')

    return im_out

def get_avg_coloured_pixel(img, left, top, width, height):
    # This function acquires the value of the average non-white pixel within the specified values.

    avg_red = 0
    avg_blue = 0
    avg_green = 0

    area = 1

    for i in range(left, min(left+width, img.shape[0])):
        for j in range(top, min(top+height, img.shape[1])):
            if img[i][j][0] < 250 or img[i][j][1] < 250 or img[i][j][2] < 250:
                avg_red += img[i][j][2]
                avg_green += img[i][j][1]
                avg_blue += img[i][j][0]
                area += 1



    avg_red /= area
    avg_blue /= area
    avg_green /= area

    return((avg_blue, avg_green, avg_red))

def get_percent_coloured(img, left, top, width, height):
    area = 0

    for i in range(left, min(left+width, img.shape[0])):
        for j in range(top, min(img.shape[1], top+height)):
            if img[i][j][0] < 250 or img[i][j][1] < 250 or img[i][j][2] < 250:
                area += 1

    total_area = float(width * height)
    return((area*100.0)/total_area)

if __name__ == "__main__":
    coords = (
        ((199, 1240), (771, 2222)),
        ((757, 1240), (1480, 1784)),
        ((771, 1815), (1835, 2210)),
        ((1835, 1705), (2344, 2222)),
        ((2344, 1222), (3130, 1798))
    )

    for i in coords:
        # tuple_out = get_data('py-testing/week_10_page_2.png', debug=False, coords = i)
        tuple_out = get_data('py-testing/week_24_page_1.png', debug=False, coords = i)

        print(tuple_out)
        break

        # print('\n===DISPLAYING PROCESSED RED CHANNEL===\n')
        # visualize_results(tuple_out[0])
        print('\n===DISPLAYING OVERALL PROCESSED BLACK TEXT===\n')
        visualize_results(tuple_out[1])

        # print(tuple_out[1]['data'])

        img = tuple_out[1]['image']

        # cv2.imshow("Image Output", img)
        # cv2.waitKey(0)

        # print(tuple_out[1]['data'])

        # input('\nEnter to continue...')
        for j in range(len(tuple_out[1]['data']['level'])):
            word = tuple_out[1]['data']['text'][j]
            left = tuple_out[1]['data']['left'][j]
            top = tuple_out[1]['data']['top'][j]
            width = tuple_out[1]['data']['width'][j]
            height = tuple_out[1]['data']['height'][j]

            if word.strip() != '':
                print('getting average color... ')
                avg_color = get_avg_coloured_pixel(img, left, top, width, height)

                print('getting percent coloured... \n')
                percent = get_percent_coloured(img, left, top, width, height)
                print('Word: {} \tAverage Colour: {} \tPercent Coloured: {} \tCoordinates: {},{}'.format(word, avg_color, percent, left, top))

                img = cv2.rectangle(img, (left, top), (left + width, top+height), (0, 0, 255), 2)

        cv2.imshow("Image Output", img)
        cv2.waitKey(0)