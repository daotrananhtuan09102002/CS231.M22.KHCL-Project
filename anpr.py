# import the necessary packages
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 7:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


def merge_contours(candidates):
    # merge the contours nearby
    LENGTH = len(candidates)
    status = np.zeros((LENGTH, 1))
    for i, cnt1 in enumerate(candidates):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(candidates[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(candidates[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    return unified
    # contours = cv2.drawContours(gray, unified, -1, (0, 255, 0), 3)
    # self.debug_imshow("Unified", contours)
def preprocessing_lp_OCR(image):
    width = 300
    height = int(300 * image.shape[0] / image.shape[1])
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    kernel = np.ones((1, 1), dtype='uint8')
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.medianBlur(image, 5)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image = 255 - image
    return image


def build_tesseract_options(psm=6):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options


class PyImageSearchANPR:
    def __init__(self, minAR=1.1, maxAR=1.6, minContourArea=1000, debug=False, save_image=False):
        # store the minimum and maximum rectangular aspect ratio
        # values along with whether or not we are in debug mode
        self.minAR = minAR
        self.maxAR = maxAR
        self.minContourArea = minContourArea
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        self.debug_imshow("Gray", gray)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        topHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
        blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        gray = cv2.add(gray, topHat)
        gray = cv2.subtract(gray, blackHat)
        self.debug_imshow("Top Hat", topHat)
        self.debug_imshow("Black hat", blackHat)
        self.debug_imshow("Subtract", gray)
        # next, find regions in the image that are light
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)
        # compute the Scharr gradient representation of the blackhat
        # image in the x-direction and then scale the result back to
        # the range [0, 255]
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)
        # blur the gradient representation, applying a closing
        # operation, and threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        # perform a series of erosions and dilations to clean up the
        # thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)
        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)
        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]


        return gray, cnts, thresh

    def locate_license_plate(self, gray, candidates,
                             clearBorder=False, debug=False):
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None
        if debug:
            imageBGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            contours = cv2.drawContours(imageBGR.copy(), candidates, -1, (0, 255, 0), 3)
            self.debug_imshow("Before merge Contours", contours)
        # merge contours
        merged_contours = merge_contours(candidates)
        if debug:
            imageBGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            contours = cv2.drawContours(imageBGR.copy(), merged_contours, -1, (0, 255, 0), 3)
            self.debug_imshow("After merge Contours", contours)
        # loop over the license plate candidate contours
        lpCnt = []
        roi_list = []
        for c in merged_contours:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # print(ar)
            # check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR and w*h > self.minContourArea:
                lpCnt.append(c)

        return lpCnt

    def character_segmentation(self, gray, lpCnt, thresh, clearborder=False, debug=False):
        # initialize the list of extracted characters
        chars = []
        # loop over the license plate contours
        for c in lpCnt:
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # extract the character ROI
            roi = gray[y:y + h, x:x + w]
            # roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                cv2.THRESH_BINARY, 11, 2)
            if clearborder:
                roi = clear_border(roi)

            # resize the character ROI to a fixed size
            cv2.imshow("ROI", roi)
            cv2.imwrite(r"outputANPR\ROI.jpg", roi)
            cv2.waitKey(0)

        # return the list of characters
        return chars
    def find_and_ocr(self, image, psm=6, clearBorder=False):
        # initialize the license plate text
        lpText = None
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        imgHue, imgSatur, gray = cv2.split(hsv)
        #
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray, candidates, thresh = self.locate_license_plate_candidates(gray)
        lpCnt = self.locate_license_plate(gray, candidates,
                                                clearBorder=clearBorder)


        # only OCR the license plate if the license plate ROI is not
        # empty
        # if lp is not None:
        #     # OCR the license plate
        #     options = build_tesseract_options(psm=psm)
        #     # cv2.imshow("License Plate 1", lp)
        #     # lp = preprocessing_lp_OCR(lp)
        #     lpText = pytesseract.image_to_string(lp, config=options)
        #     self.debug_imshow("License Plate", lp)
        #     cv2.imshow("License Plate", lp)
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        img = image.copy()
        if lpCnt is None:
            cv2.imshow("Output", img)
            return None, None
        count = 0

        ROI_list = []
        for i in lpCnt:
            # hien thi vung bien so chinh xac nhat
            box = cv2.boxPoints(cv2.minAreaRect(i))
            box = box.astype("int")
            cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
            cv2.imshow("Output", img)

           # cat anh de doc chu
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.imshow('Output', img)
            # doc chu tu anh gray
            ROI = gray[y:y + h, x:x + w]
            ROI = imutils.resize(ROI, width=200)
            ROI_list.append(ROI)
            count += 1
            # cv2.imshow(f"{count}", ROI)
        # cv2.waitKey(0)
        # char = self.character_segmentation(gray, lpCnt, thresh, clearborder=clearBorder)

        return lpText, lpCnt, ROI_list
