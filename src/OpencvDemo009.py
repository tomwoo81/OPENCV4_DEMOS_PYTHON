#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像色彩空间转换
def OpencvDemo009():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/cat.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("bgr", cv.WINDOW_AUTOSIZE)
    cv.imshow("bgr", src)
    
    # BGR to HSV
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    
    # BGR to YUV
    yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    
    # BGR to YCrCb
    ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", ycrcb)
    
    src2 = cv.imread("images/greenback.png")
    if src2 is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.imshow("src2", src2)
    
    hsv = cv.cvtColor(src2, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (35, 43, 46), (77, 255, 255))
    cv.imshow("mask", mask)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo009()

# end of file
