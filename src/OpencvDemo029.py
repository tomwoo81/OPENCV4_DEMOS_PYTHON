#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 快速的图像边缘滤波算法
def OpencvDemo029():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/example.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    freq = cv.getTickFrequency()
    start = cv.getTickCount()
    
    dst = cv.edgePreservingFilter(src, sigma_s=100, sigma_r=0.4, flags=cv.RECURS_FILTER)
    
    end = cv.getTickCount()
    time = (end - start) / (freq / 1000)
    logging.debug("time consumed: %.3f ms", time)
    
    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "filtered image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("edge-preserving filtering", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo029()

# end of file
