import cv2
import os
import time
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

nPlateCascade = cv2.CascadeClassifier(r'G:\My Drive\automatic-license-number-plate-recognition\ANPR_Ver.2\resources\haarcascade_motor_license_plate.xml')
# nPlateCascade = cv2.CascadeClassifier('resources/haarcascade_licence_plate_rus_16stages.xml')

minArea = 150


def detect_license_plate(img):
    width = 600
    height = int(600 * img.shape[0] / img.shape[1])
    plate = None

    img = cv2.resize(img, (width, height))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_plates = nPlateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in box_plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
        plate = img[y:y+h, x:x + w]

    return img, plate


start = time.time()
i = 0
count = 0
for file in os.listdir(r"G:\My Drive\automatic-license-number-plate-recognition\GreenParking"):
    if file.endswith(".jpg") and i < 20:


        i += 1
        img = cv2.imread("G:/My Drive/automatic-license-number-plate-recognition/GreenParking/" + file)
        img, plate = detect_license_plate(img)
        print('------------------------------------------------------')
        print(f'Processing image: {i}')
        if plate is not None:
            count += 1
            cv2.imshow('Plate', plate)
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            char = reader.readtext(plate_gray, detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ-.')
            print('Number plate is:')
            print('\n'.join(char))
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print('Number of image detected: ', count)
print('Percent of detected image: {:.2f}'.format(count / i * 100))
print('Time: ', time.time() - start)