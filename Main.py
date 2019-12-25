import numpy as np
import cv2
import imutils
from transform import *
from skimage.filters import threshold_local


def main():

    image=cv2.imread('receipt.jpg')

    ratio=image.shape[0]/500.0
    orignal=image.copy()
    image=imutils.resize(image,height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 75, 200)


    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    cnts=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]


    for c in cnts:
        preimeter=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*preimeter,True)
        print(len(approx))
        if len(approx)==4:
            document=approx
            #break

    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [document], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)

    pic=four_points_transform(orignal,document.reshape(4,2)*ratio)

    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic=cv2.adaptiveThreshold(pic,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,10)




    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orignal, height=650))
    cv2.imshow("Scanned", imutils.resize(pic, height=650))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()