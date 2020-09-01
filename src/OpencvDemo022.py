#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像均值与高斯模糊
def OpencvDemo022():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/snow.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    dst1 = cv.blur(src, (5, 5))
    dst2 = cv.GaussianBlur(src, (5, 5), sigmaX=15)
    dst3 = cv.GaussianBlur(src, (0, 0), sigmaX=15)
    cv.imshow("blur ksize=5", dst1)
    cv.imshow("gaussian ksize=5", dst2)
    cv.imshow("gaussian sigmax=15", dst3)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo022()

# end of file
