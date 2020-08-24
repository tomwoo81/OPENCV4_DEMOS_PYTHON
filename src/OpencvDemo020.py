#!/usr/bin/env python3
#coding = utf-8

import logging
from matplotlib import pyplot as plt
import cv2 as cv

# 图像直方图反向投影
def OpencvDemo020():
    logging.basicConfig(level=logging.DEBUG)
    
    target = cv.imread("images/target.png")
    sample = cv.imread("images/sample.png")
    if (target is None) or (sample is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.namedWindow("target", cv.WINDOW_AUTOSIZE)
    cv.imshow("target", target)
    cv.imshow("sample", sample)
    
    hist2d_demo(target)
    hist2d_demo(sample)
    
    back_projection_demo(target, sample)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    
    dst = cv.resize(hist, (400, 400))
    cv.imshow("hist", dst)
    
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()

def back_projection_demo(image, sample):
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    
    roiHist = cv.calcHist([sample_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    
    backProj = cv.calcBackProject([image_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow("BackProjection", backProj)

if __name__ == "__main__":
    OpencvDemo020()

# end of file
