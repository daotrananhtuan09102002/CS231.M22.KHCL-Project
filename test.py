import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def build_tesseract_options(psm=6):
    # tell Tesseract to only OCR alphanumeric characters
    # alphanumeric = "ABCDEFGHKLMNPRSTUVXYZ0123456789"
    alphanumeric = "0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options


def preprocessing_lp_OCR(image):
    # width = 150
    # height = int(150 * image.shape[0] / image.shape[1])
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    kernel = np.ones((5, 5), dtype='uint8')
    # image = ~image
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.medianBlur(image, 3)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return image


config = build_tesseract_options(10)

# for i in os.listdir("./outputANPR2/"):
#     img = cv2.imread(os.path.join(r"./outputANPR2", i), 0)
#     # print(img.shape)
#     img = preprocessing_lp_OCR(img)
#     cv2.imshow('img', img)
#     print(pytesseract.image_to_string(img, config=config))
#     cv2.waitKey(0)

# img = cv2.imread(r"./outputANPR2/842.jpg", 0)
# cv2.imshow('img', img)
# print(pytesseract.image_to_string(img, config=config))
# cv2.waitKey(0)

img = cv2.imread(r"./outputANPR2/Untitled4.png",0)
cv2.imshow('img', img)
h, w = img.shape
# img = preprocessing_lp_OCR(img)
# print(pytesseract.image_to_string(img, config=config))
img = cv2.GaussianBlur(img, (1, 1), 0)
img = cv2.medianBlur(img, 1)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
boxes = pytesseract.image_to_boxes(img, config=config) # also include any config options you use
new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    print(b[0])
    img = cv2.rectangle(new_img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', new_img)
cv2.waitKey(0)
# print(pytesseract.get_tesseract_version())