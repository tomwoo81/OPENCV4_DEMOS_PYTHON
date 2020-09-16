#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# TRIANGLE二值寻找算法
def OpencvDemo043():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/lena.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 自动阈值分割 TRIANGLE
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    t, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    logging.debug("threshold - triangle: {:.0f}".format(t))
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "binary image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("binarize - triangle", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo043()

# end of file
