#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# OpenCV自定义的滤波器
def OpencvDemo030():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    blur_op = np.ones([5, 5], dtype=np.float32) / 25.
    sharpen_op = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], np.float32)
    gradient_op = np.array([[ 1,  0], [ 0, -1]], dtype=np.float32)
    
    dst1 = cv.filter2D(src, -1, blur_op)
    dst2 = cv.filter2D(src, -1, sharpen_op)
    dst3 = cv.filter2D(src, cv.CV_32F, gradient_op)
    dst3 = cv.convertScaleAbs(dst3)
    
    cv.imshow("blur=5x5", dst1)
    cv.imshow("sharpen=3x3", dst2)
    cv.imshow("gradient=2x2", dst3)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo030()

# end of file
