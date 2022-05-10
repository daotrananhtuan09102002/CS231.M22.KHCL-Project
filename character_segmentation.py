from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv

MIN_RATIO_AREA = 0.02
MAX_RATIO_AREA = 0.07
MIN_RATIO = 2
MAX_RATIO = 5.5


def KMeans_(img, n_clusters=3):
    nrow, ncol = img.shape
    g = img.reshape(nrow * ncol, -1)
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(g)
    t = k_means.cluster_centers_[k_means.labels_]
    img_res = t.reshape(nrow, ncol)
    img_res = img_res.astype(np.uint8)
    return img_res


class character_segmentation:
    def __init__(self, img, n_clusters=3):
        self.img = img
        self.n_clusters = n_clusters

    def segment(self):
        seg_img = KMeans_(self.img, self.n_clusters)
        area = seg_img.shape[0] * seg_img.shape[1]
        ret, thresh = cv.threshold(seg_img, 100, 255, cv.THRESH_BINARY)
        blur = cv.GaussianBlur(thresh, (5, 5), 0)
        im_bw = cv.Canny(blur, 10, 200)

        cv.imshow("Canny", im_bw)

        img_BGR = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)

        contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:12]
        cnts = cv.drawContours(img_BGR.copy(), contours, -1, (0, 255, 0), 3)
        cv.imshow("Contour", cnts)

        new_contours = []
        img_BGR = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        for c in contours:
            (x, y, w, h) = cv.boundingRect(c)
            # print('Cnt area:', w * h, 'ratio', h / w)
            if MIN_RATIO_AREA * area <= w * h <= MAX_RATIO_AREA * area \
                    and MIN_RATIO <= h / w <= MAX_RATIO:
                new_contours.append(c)

        for c in new_contours:
            # box = cv.minAreaRect(c)
            # box = cv.boxPoints(box)
            # box = np.int0(box)
            # cv.drawContours(img_BGR, [box], 0, (0, 0, 255), 2)

            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(img_BGR, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("Segmented", img_BGR)
