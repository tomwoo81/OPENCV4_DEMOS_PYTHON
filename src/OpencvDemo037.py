#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像金字塔
def OpencvDemo037():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("pyramid down - level 0", cv.WINDOW_AUTOSIZE)
    cv.imshow("pyramid down - level 0", src)
    
    # pyramidDown(src)
    pyramidUp(pyramidDown(src))
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def pyramidDown(image, level=3):
    temp = image.copy()
    pyramid = []
    
    for i in range(level):
        dst = cv.pyrDown(temp)
        cv.imshow("pyramid down - level {:d}".format(i + 1), dst)
        
        temp = dst.copy()
        pyramid.append(temp)
    
    return pyramid

def pyramidUp(pyramid):
    for i in range(len(pyramid) - 1, -1, -1):
        dst = cv.pyrUp(pyramid[i])
        cv.imshow("pyramid up - level {:d}".format(i), dst)

if __name__ == "__main__":
    OpencvDemo037()

# end of file
