# import the necessary packages
import numpy as np
import imutils
import cv2


class PyImageSearchANPR:
    def __init__(self, minAR=1.1, maxAR=1.6, minContourArea=1000, debug=False):
        # khởi tạo tham số
        self.minAR = minAR
        self.maxAR = maxAR
        self.minContourArea = minContourArea
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        # nếu chọn chế độ debug thì hiển thị ảnh
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

        # tăng độ tương phản cho ảnh xám
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        topHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
        blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        gray = cv2.add(gray, topHat)
        gray = cv2.subtract(gray, blackHat)
        self.debug_imshow("Top Hat", topHat)
        self.debug_imshow("Black hat", blackHat)
        self.debug_imshow("Subtract", gray)

        # next, find regions in the image that are light
        # tìm những vùng trong ảnh có màu sắc trắng
        squareKern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
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
        thresh = cv2.erode(thresh, None, iterations=3) # 3
        thresh = cv2.dilate(thresh, None, iterations=3) # 3
        self.debug_imshow("Grad Erode/Dilate", thresh)
        # take the bitwise AND between the threshold result and the
        # light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        self.debug_imshow("Grad Thresh & Light", thresh)
        thresh = cv2.dilate(thresh, None, iterations=3) # 3
        thresh = cv2.erode(thresh, None, iterations=1)  # 1
        self.debug_imshow("Final", thresh, waitKey=True)
        # find contours in the thresholded image and sort them by
        # their size in descending order, keeping only the largest
        # ones
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        return gray, cnts, thresh

    def locate_license_plate(self, gray, candidates):
        lpCnt = []
        for c in candidates:
            # tính toán kích thước của contour
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # print(ar)

            # kiểm tra kích thước contour có phù hợp không
            if self.minAR <= ar <= self.maxAR and w * h > self.minContourArea:
                lpCnt.append(c)

        return lpCnt

    def find_and_ocr(self, image):
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray, candidates, thresh = self.locate_license_plate_candidates(gray)
        lpCnt = self.locate_license_plate(gray, candidates)
        if len(lpCnt) == 0:
            return [], []

        ROI_list = []
        for i in lpCnt:
            # hien thi vung bien so chinh xac nhat
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
            # cv2.imshow("Output", img)

            # cat anh de doc chu
            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            angle = rect[2]
            if angle > 45:
                angle -= 90

            # xoay ảnh
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # Size of the upright rectangle bounding the rotated rectangle
            size = (x2 - x1, y2 - y1)
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
            # Cropped upright rectangle
            cropped = cv2.getRectSubPix(gray, size, center)
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = H if H > W else W
            croppedH = H if H < W else W
            # Final cropped & rotated rectangle
            ROI = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
            ROI = imutils.resize(ROI, width=200)
            ROI_list.append(ROI)

        return lpCnt, ROI_list
