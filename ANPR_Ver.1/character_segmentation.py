from sklearn.cluster import KMeans
import numpy as np
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

MIN_RATIO_AREA = 0.015
MAX_RATIO_AREA = 0.07
MIN_RATIO = 2
MAX_RATIO = 6


def KMeans_(img, n_clusters=3):
    nrow, ncol = img.shape
    g = img.reshape(nrow * ncol, -1)
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(g)
    t = k_means.cluster_centers_[k_means.labels_]
    img_res = t.reshape(nrow, ncol)
    img_res = img_res.astype(np.uint8)
    return img_res


class character_segmentation:
    def __init__(self, n_clusters=3, debug=False):
        self.n_clusters = n_clusters
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    def segment(self, img):
        # read text from license plate image with segmentation
        self.debug_imshow("Original", img)
        height, width = img.shape
        # use KMeans to segment image into 3 clusters
        seg_img = KMeans_(img, self.n_clusters)
        area = seg_img.shape[0] * seg_img.shape[1]
        seg_img = seg_img.astype(np.uint8)
        self.debug_imshow("seg_img", seg_img)
        # binarize image
        ret, thresh = cv2.threshold(seg_img, 100, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh, (5, 5), 0)
        im_bw = cv2.Canny(blur, 10, 200)
        self.debug_imshow('im_bw', im_bw)

        img_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        if self.debug:
            cnts = cv2.drawContours(img_BGR.copy(), contours, -1, (0, 255, 0), 3)
            self.debug_imshow('Contours', cnts)

        new_contours = []

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            # print('Cnt area:', w * h, 'ratio', h / w)
            # remain only contours with area between MIN_RATIO_AREA and MAX_RATIO_AREA
            # and ratio between MIN_RATIO and MAX_RATIO
            if MIN_RATIO_AREA * area <= w * h <= MAX_RATIO_AREA * area \
                    and MIN_RATIO <= h / w <= MAX_RATIO:
                new_contours.append(c)

        chars = []
        X, Y = [], []
        # segment image into characters
        for c in new_contours:
            (x, y, w, h) = cv2.boundingRect(c)

            # padding to help with OCR accuracy
            x_cut = x - 5 if x - 5 > 0 else 0
            y_cut = y - 5 if y - 5 > 0 else 0
            w_cut = w + 5 if x + w + 10 < width else w
            h_cut = h + 5 if y + h + 10 < height else h

            if x_cut == 0 or x + w_cut == width:
                continue
            chars.append(img[y_cut:y + h_cut, x_cut:x + w_cut])
            X.append(x)
            Y.append(y)

        s_top, s_bottom = [], []
        chars_top = []
        chars_bottom = []

        # divide characters into top and bottom by y-coordinate
        for i, c in enumerate(chars):
            if Y[i] < height / 2 - 30:
                chars_top.append([c, X[i], Y[i]])
            else:
                chars_bottom.append([c, X[i], Y[i]])

        # divide top and bottom characters into two groups by x-coordinate
        chars_top = sorted(chars_top, key=lambda x: x[1])
        chars_bottom = sorted(chars_bottom, key=lambda x: x[1])

        #
        if len(chars_top) < 4:
            for i in range(len(chars_top)):
                h, w = chars_top[i][0].shape[:2]
                s = reader.recognize(chars_top[i][0], detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ')
                if s is not None and self.debug:
                    cv2.putText(img_BGR, *s, (chars_top[i][1] + 8, chars_top[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
                    cv2.rectangle(img_BGR, (chars_top[i][1], chars_top[i][2]),
                                  (chars_top[i][1] + w - 5, chars_top[i][2] + h - 5), (0, 255, 0), 2)

                    self.debug_imshow('char', chars_top[i][0], waitKey=True)

                s_top.append(*s)
        else:
            for i in range(len(chars_top)):
                h, w = chars_top[i][0].shape[:2]
                if i == 2:
                    s = reader.recognize(chars_top[i][0], detail=0, allowlist='ABCDEFGHKLMNPRSTUVXYZ')
                else:
                    s = reader.recognize(chars_top[i][0], detail=0, allowlist='0123456789')
                if s is not None and self.debug:
                    cv2.putText(img_BGR, *s, (chars_top[i][1] + 8, chars_top[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
                    cv2.rectangle(img_BGR, (chars_top[i][1], chars_top[i][2]),
                                  (chars_top[i][1] + w - 5, chars_top[i][2] + h - 5), (0, 255, 0), 2)

                    self.debug_imshow('char', chars_top[i][0], waitKey=True)

                s_top.append(*s)

        for i in range(len(chars_bottom)):
            h, w = chars_bottom[i][0].shape[:2]
            s = reader.recognize(chars_bottom[i][0], detail=0, allowlist='0123456789')
            if s is not None and self.debug:
                cv2.putText(img_BGR, *s, (chars_bottom[i][1] + 8, chars_bottom[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)
                cv2.rectangle(img_BGR, (chars_bottom[i][1], chars_bottom[i][2]),
                              (chars_bottom[i][1] + w - 5, chars_bottom[i][2] + h - 5), (0, 255, 0), 2)

                self.debug_imshow('char', chars_bottom[i][0], waitKey=True)

            s_bottom.append(*s)

        self.debug_imshow('Result', img_BGR)
        return ''.join(s_top) + '\n' + ''.join(s_bottom)

    def no_segment(self, img):
        # read text from license plate image without segmentation
        s = reader.readtext(img, detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ.-')
        for i in range(len(s)):
            try:
                s[i] = s[i].replace('.', '')
                s[i] = s[i].replace('-', '')
            except:
                continue

        return '\n'.join(s)
