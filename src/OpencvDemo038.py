#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 拉普拉斯金字塔
def OpencvDemo038():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("pyramid down - level 0", cv.WINDOW_AUTOSIZE)
    cv.imshow("pyramid down - level 0", src)
    
    # pyramidDown(src)
    laplacianPyramid(src, pyramidDown(src))
    
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

def laplacianPyramid(image, pyramid):
    for i in range(len(pyramid) - 1, -1, -1):
        if i > 0:
            h, w = pyramid[i - 1].shape[:2]
            dst = cv.pyrUp(pyramid[i], dstsize=(w, h))
            dst = cv.subtract(pyramid[i - 1], dst)
        else:
            h, w = image.shape[:2]
            dst = cv.pyrUp(pyramid[i], dstsize=(w, h))
            dst = cv.subtract(image, dst)
        
        dst += 128
        cv.imshow("laplacian pyramid - level {:d}".format(i), dst)

if __name__ == "__main__":
    OpencvDemo038()

# end of file
