# import the necessary packages
from anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2
from character_segmentation import character_segmentation
import time


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
    imagePaths = sorted(list(paths.list_images(args["input"])))
    # loop over all image paths in the input directory
    count = 0
    start = time.time()
    for _, imagePath in enumerate(imagePaths, start=1):
        # load the input image from disk and resize it
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        # apply automatic license plate recognition
        lpCnt, ROI_list = anpr.find_and_ocr(image)
        print('----------------------------------------------')
        print(f'Processing image: {_}')
        if len(lpCnt) == 0:
            print("Can't detect license plate")
            cv2.imshow(f"Output{_}", image)
            # cv2.waitKey(0)
            continue
        flag = False
        for (i, r) in enumerate(ROI_list):
            char = seg_anpr.segment(r)
            if len(char) > 2:
                count += 1

                x, y, w, h = cv2.boundingRect(lpCnt[i])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, '{}'.format(char.replace('\n', '-')),
                             (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.imshow(f"Output_{_}", image)
                print('Characters detected:\n' + char)
                flag = True

        cv2.waitKey(0)
        if flag is False:
            print('No characters detected')
        cv2.destroyAllWindows()

    print(f'Total time: {time.time() - start}')
    print('Detected {} license plates'.format(count))
    print('Number of images: {}'.format(len(imagePaths)))
    print('Percentage of detected license plates: {:.2f}'.format(count / len(imagePaths)))


if __name__ == "__main__":
    args = parse_args()
    main(args)
