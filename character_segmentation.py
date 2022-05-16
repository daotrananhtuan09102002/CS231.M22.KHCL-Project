from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def build_tesseract_options(psm=10):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHKLMNPRSTUVXYZ0123456789"
    # alphanumeric = "0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options


def preprocessing_lp_OCR(image):
    width = 150
    height = int(150 * image.shape[0] / image.shape[1])
    image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    kernel = np.ones((5, 5), dtype='uint8')
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.medianBlur(image, 3)
    image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    return image


MIN_RATIO_AREA = 0.02
MAX_RATIO_AREA = 0.07
MIN_RATIO = 2
MAX_RATIO = 5.5
config = build_tesseract_options(10)


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
            cv.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv.waitKey(0)

    def segment(self, img):
        seg_img = KMeans_(img, self.n_clusters)
        area = seg_img.shape[0] * seg_img.shape[1]
        seg_img = seg_img.astype(np.uint8)
        ret, thresh = cv.threshold(seg_img, 100, 255, cv.THRESH_BINARY)
        blur = cv.GaussianBlur(thresh, (5, 5), 0)
        im_bw = cv.Canny(blur, 10, 200)
        # im_bw = cv.morphologyEx(im_bw, cv.MORPH_CLOSE, kernel = (3, 3), iterations=2)
        cv.imshow("Canny", im_bw)

        img_BGR = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        contours, hierarchy = cv.findContours(im_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:15]
        cnts = cv.drawContours(img_BGR.copy(), contours, -1, (0, 255, 0), 3)
        cv.imshow("Contour", cnts)

        new_contours = []
        img_BGR = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for c in contours:
            (x, y, w, h) = cv.boundingRect(c)
            # print('Cnt area:', w * h, 'ratio', h / w)
            if MIN_RATIO_AREA * area <= w * h <= MAX_RATIO_AREA * area \
                    and MIN_RATIO <= h / w <= MAX_RATIO:
                new_contours.append(c)

        chars = []
        X, Y = [], []
        for c in new_contours:
            (x, y, w, h) = cv.boundingRect(c)
            chars.append(seg_img[y:y + h + 5, x:x + w + 5])
            X.append(x)
            Y.append(y)
            cv.rectangle(img_BGR, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)

        vis_list = []
        for j, c in enumerate(chars):
            list_img = []
            h, w = c.shape[:2]
            base_size = (h + 50, w + 50)
            for i in range(3):
                base = np.full(base_size, 255, dtype=int)
                cv.rectangle(base, (0, 0), (w + 50, h + 50), (255, 255, 255), 30)
                base[25:h + 25, 25:w + 25] = c
                list_img.append(base)
            vis = np.concatenate((list_img[0], list_img[1]), axis=1)
            vis = np.concatenate((vis, list_img[2]), axis=1)
            vis = vis.astype(np.uint8)
            vis = preprocessing_lp_OCR(vis)
            s = pytesseract.image_to_string(vis, config=config)
            if len(s) > 0:
                s_max = max(s, key=s.count)
                cv.putText(img_BGR, s_max, (X[j] + 10, Y[j] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv.imshow('merge', vis)
            cv.waitKey(0)
        cv.imshow("Segmented", img_BGR)
        return vis_list
