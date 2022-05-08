import pytesseract

from anpr import build_tesseract_options, preprocessing_lp_OCR
import cv2
import numpy as np


img = cv2.imread('license_plate/008.png')
img = cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
cv2.imshow('resize', img)
config = build_tesseract_options(6)
print(pytesseract.image_to_string(img, config=config))
img = preprocessing_lp_OCR(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cv2.imshow('preprocessing', img)
print(pytesseract.image_to_string(img, config=config))
cv2.waitKey(0)


