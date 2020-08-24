#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 像素归一化
def OpencvDemo011():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    # 转换为浮点数类型数组
    gray = np.float32(gray)
    print(gray)
    
    # scale and shift by NORM_MINMAX
    dst = np.zeros(gray.shape, dtype=np.float32)
    cv.normalize(gray, dst=dst, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
    print(dst)
    cv.imshow("NORM_MINMAX", np.uint8(dst*255))
    
    # scale and shift by NORM_INF
    dst = np.zeros(gray.shape, dtype=np.float32)
    cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_INF)
    print(dst)
    cv.imshow("NORM_INF", np.uint8(dst*255))
    
    # scale and shift by NORM_L1
    dst = np.zeros(gray.shape, dtype=np.float32)
    cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L1)
    print(dst)
    cv.imshow("NORM_L1", np.uint8(dst*10000000))
    
    # scale and shift by NORM_L2
    dst = np.zeros(gray.shape, dtype=np.float32)
    cv.normalize(gray, dst=dst, alpha=1.0, beta=0, norm_type=cv.NORM_L2)
    print(dst)
    cv.imshow("NORM_L2", np.uint8(dst*10000))
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo011()

# end of file
