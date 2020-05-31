from cv2 import cv2 
import os 
import numpy as np 


def grayingImage(testImage):
    toGrayImage= cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
    return toGrayImage 


def adjust_gamma(image, gamma=1.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def HistogramEqualization(img):
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
 
    return hist_equalization_result
    
def DoG(fn):
    
    # fn_no_ext = fn.split('.')[0]
    # outputFile = fn_no_ext+'DoG.jpg'
    # #read the input file
    img = cv2.imread(str(fn))

    #run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur5 = cv2.GaussianBlur(fn,(9,9),1)
    blur3 = cv2.GaussianBlur(fn,(3,3),1)

    #write the results of the previous step to new files
    cv2.imwrite('3x3.jpg', blur3)
    cv2.imwrite('5x5.jpg', blur5)

    DoGim = blur5 - blur3
    return DoGim