#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像梯度–Sobel算子
def OpencvDemo031():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    grad_x = cv.Sobel(src, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(src, cv.CV_32F, 0, 1)
    grad_x = cv.convertScaleAbs(grad_x)
    grad_y = cv.convertScaleAbs(grad_y)
    
    cv.imshow("gradient - sobel operator (x)", grad_x)
    cv.imshow("gradient - sobel operator (y)", grad_y)
    
    dst = cv.add(grad_x, grad_y, dtype=cv.CV_16S)
    dst = cv.convertScaleAbs(dst)
    
    cv.imshow("gradient - sobel operator", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo031()

# end of file
