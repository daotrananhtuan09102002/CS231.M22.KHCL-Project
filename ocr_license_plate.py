# import the necessary packages
from anpr import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2
from character_segmentation import character_segmentation

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
                help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=6,
                help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help="whether or not to show additional visualizations")
ap.add_argument("-s", "--save", type=int, default=-1,
                help="save output or not")
args = vars(ap.parse_args())

# initialize our ANPR class
anpr = PyImageSearchANPR(debug=args["debug"] > 0, save_image=args["save"] > 0)


# grab all image paths in the input directory
imagePaths = sorted(list(paths.list_images(args["input"])))[100:120]
# loop over all image paths in the input directory

count = 0
for i, imagePath in enumerate(imagePaths, start=1):
    # load the input image from disk and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    # apply automatic license plate recognition
    lpText, lpCnt, ROI_list = anpr.find_and_ocr(image, psm=args["psm"],
                                        clearBorder=args["clear_border"] > 0)
    print(f'Processing image: {i}')
    # only continue if the license plate was successfully OCR'd
    # if lpText is not None and lpCnt is not None:
    if len(lpCnt) > 0:
        count += 1
        # for j, x in enumerate(ROI_list,start=1):
        #     cv2.imwrite(f'outputANPR2/{count}_{j}.jpg', x)
        # loop over the contours
        for (j, r) in enumerate(ROI_list, start=1):
            seg_img = character_segmentation(r, n_clusters = 3)
            seg_img.segment()
            cv2.waitKey(0)
    cv2.destroyAllWindows()

print('Detected {} license plates'.format(count))
print('Number of images: {}'.format(len(imagePaths)))
print('Percentage of detected license plates: {:.2f}'.format(count/len(imagePaths)))
