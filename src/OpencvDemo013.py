#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像翻转
def OpencvDemo013():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    # X Flip 倒影
    dst1 = cv.flip(src, 0);
    cv.imshow("x-flip", dst1);
    
    # Y Flip 镜像
    dst2 = cv.flip(src, 1);
    cv.imshow("y-flip", dst2);
    
    # XY Flip 对角
    dst3 = cv.flip(src, -1);
    cv.imshow("xy-flip", dst3);
    
    # custom y-flip
    h, w, ch = src.shape
    dst = np.zeros(src.shape, src.dtype)
    for row in range(h):
        for col in range(w):
            b, g, r = src[row, col]
            dst[row, w - col - 1] = [b, g, r]
    cv.imshow("custom-y-flip", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo013()

# end of file
