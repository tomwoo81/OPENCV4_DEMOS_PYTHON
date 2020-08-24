#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像像素的算术操作
def OpencvDemo005():
    logging.basicConfig(level=logging.DEBUG)
    
    src1 = cv.imread("images/LinuxLogo.jpg")
    src2 = cv.imread("images/WindowsLogo.jpg")
    if (src1 is None) or (src2 is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("input1", src1)
    cv.imshow("input2", src2)
    
    h, w, ch = src1.shape
    logging.debug("h: %u, w: %u, ch: %u", h, w, ch)
    
    add_result = np.zeros(src1.shape, src1.dtype);
    cv.add(src1, src2, add_result);
    cv.imshow("add_result", add_result);
    
    sub_result = np.zeros(src1.shape, src1.dtype);
    cv.subtract(src1, src2, sub_result);
    cv.imshow("sub_result", sub_result);
    
    mul_result = np.zeros(src1.shape, src1.dtype);
    cv.multiply(src1, src2, mul_result);
    cv.imshow("mul_result", mul_result);
    
    div_result = np.zeros(src1.shape, src1.dtype);
    cv.divide(src1, src2, div_result);
    cv.imshow("div_result", div_result);
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo005()

# end of file
