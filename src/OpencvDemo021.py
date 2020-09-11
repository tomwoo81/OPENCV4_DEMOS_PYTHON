#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像卷积操作
def OpencvDemo021():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/lena.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    dst = custom_blur(src)
    cv.imshow("blur", dst)
    cv.imwrite("output/blur.png", dst)
    
    result = cv.blur(src, (15, 15))
    cv.imshow("result", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def custom_blur(src):
    h, w, ch = src.shape
    logging.debug("h: {:d}, w: {:d}, ch: {:d}".format(h, w, ch))
    
    dst = np.copy(src)
    for row in range(1, h-1, 1):
        for col in range(1, w-1, 1):
            v1 = np.int32(src[row-1, col-1])
            v2 = np.int32(src[row-1, col])
            v3 = np.int32(src[row-1, col+1])
            v4 = np.int32(src[row, col-1])
            v5 = np.int32(src[row, col])
            v6 = np.int32(src[row, col+1])
            v7 = np.int32(src[row+1, col-1])
            v8 = np.int32(src[row+1, col])
            v9 = np.int32(src[row+1, col+1])
            
            b = v1[0] + v2[0] + v3[0] + v4[0] + v5[0] + v6[0] + v7[0] + v8[0] + v9[0];
            g = v1[1] + v2[1] + v3[1] + v4[1] + v5[1] + v6[1] + v7[1] + v8[1] + v9[1];
            r = v1[2] + v2[2] + v3[2] + v4[2] + v5[2] + v6[2] + v7[2] + v8[2] + v9[2];
            
            dst[row, col] = [b//9, g//9, r//9]
    
    return dst

if __name__ == "__main__":
    OpencvDemo021()

# end of file
