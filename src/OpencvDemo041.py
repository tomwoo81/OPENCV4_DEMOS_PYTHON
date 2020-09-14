#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# OpenCV中的基本阈值操作
def OpencvDemo041():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    # 转换为灰度图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    t = 127
    
    """
    THRESH_BINARY = 0,
    THRESH_BINARY_INV = 1.
    THRESH_TRUNC = 2,
    THRESH_TOZERO = 3,
    THRESH_TOZERO_INV = 4
    """
    for i in range(5):
        _, dst = cv.threshold(gray, t, 255, i)
        cv.imshow("thresholding - type {:d}".format(i), dst)
    
    ret, dst = cv.threshold(gray, t, 255, cv.THRESH_OTSU)
    logging.debug("threshold - otsu: {:.0f}".format(ret))
    cv.imshow("thresholding - otsu", dst)
    
    ret, dst = cv.threshold(gray, t, 255, cv.THRESH_TRIANGLE)
    logging.debug("threshold - triangle: {:.0f}".format(ret))
    cv.imshow("thresholding - triangle", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo041()

# end of file
