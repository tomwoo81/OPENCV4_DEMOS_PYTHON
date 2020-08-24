#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像像素的逻辑操作
def OpencvDemo007():
    logging.basicConfig(level=logging.DEBUG)
    
    # create image one
    src1 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
    src1[100:200, 100:200, 1] = 255
    src1[100:200, 100:200, 2] = 255
    cv.imshow("input1", src1)
    
    # create image two
    src2 = np.zeros(shape=[400, 400, 3], dtype=np.uint8)
    src2[150:250, 150:250, 2] = 255
    cv.imshow("input2", src2)
    
    dst1 = cv.bitwise_and(src1, src2)
    dst2 = cv.bitwise_xor(src1, src2)
    dst3 = cv.bitwise_or(src1, src2)
    
    cv.imshow("dst1", dst1)
    cv.imshow("dst2", dst2)
    cv.imshow("dst3", dst3)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    dst = cv.bitwise_not(src)
    cv.imshow("dst", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo007()

# end of file
