import cv2 
import pytesseract
from pytesseract import Output

def get_data(path: str, coords: tuple, conf=r'--oem 3 --psm 6', debug=False):
    raw_img = cv2.imread(path)

    crop_img = raw_img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]

    cv2.imshow("Cropped image", crop_img)
    cv2.waitKey(0)


    s = pytesseract.image_to_string(crop_img, config=conf)
    d = pytesseract.image_to_data(crop_img, output_type=Output.DICT, config=conf)

    return(crop_img, d, s)

img, data, string = get_data('py-testing/week_10_page_1_cropped_png.png', ((917, 517), (1355, 868)) )

print('Data lengths: {} {} {}'.format(len(data['level']),len(data['conf']),len(data['text']) ))

line_1 = []

for i in range(0, len(data['level'])):
    if(data['level'][i] == 5):
        line_1 += [data['text'][i]]
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print(data['level'][i])

print(line_1)

cv2.imshow('img', img)
cv2.waitKey(0)
