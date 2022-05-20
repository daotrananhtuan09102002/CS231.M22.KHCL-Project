# import the necessary packages
from anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2
from character_segmentation import character_segmentation
import numpy as np

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input directory of images")
    ap.add_argument("-d", "--debug", type=int, default=-1,
                    help="whether or not to show additional visualizations")
    args = vars(ap.parse_args())

    return args


def main(args):
    # initialize our ANPR class
    anpr = PyImageSearchANPR(debug=args["debug"] > 0)
    seg_anpr = character_segmentation(n_clusters=3, debug=args["debug"] > 0)

    # grab all image paths in the input directory
    imagePaths = sorted(list(paths.list_images(args["input"])))[0:10]
    # loop over all image paths in the input directory
    count = 0
    for _, imagePath in enumerate(imagePaths, start=1):
        # load the input image from disk and resize it
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        # apply automatic license plate recognition
        lpCnt, ROI_list = anpr.find_and_ocr(image)
        print(f'Processing image: {_}')
        if len(lpCnt) == 0:
            print("Can't detect license plate")
            cv2.imshow("Output", image)
            continue
        flag = False
        for (i, r) in enumerate(ROI_list):
            char = seg_anpr.segment(r)
            if len(char) > 2:
                count += 1
                rect = cv2.minAreaRect(lpCnt[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
                cv2.imshow(f"Output_{_}", image)
                print('Characters detected:\n' + char)
                flag = True
        cv2.waitKey(0)
        if flag is False:
            print('No characters detected')
        cv2.destroyAllWindows()

    print('Detected {} license plates'.format(count))
    print('Number of images: {}'.format(len(imagePaths)))
    print('Percentage of detected license plates: {:.2f}'.format(count / len(imagePaths)))


if __name__ == "__main__":
    args = parse_args()
    main(args)
