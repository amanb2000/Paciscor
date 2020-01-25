import cv2 
import pytesseract

img = cv2.imread('superscript-2.png')

# Adding custom options

custom_config = r'--oem 3 --psm 6'
# pytesseract.image_to_string(img, config=custom_config)

a = pytesseract.image_to_string(img, config=custom_config)

print(a)

