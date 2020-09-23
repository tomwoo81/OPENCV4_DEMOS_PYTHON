#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像二值化与去噪
def OpencvDemo045():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/coins.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化
    dst1 = threshold_1(src)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst1 = cv.cvtColor(dst1, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst1
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "binary image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("binarize - otsu", result)
    
    # 高斯模糊去噪声与二值化
    dst2 = threshold_2(src)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst2 = cv.cvtColor(dst2, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst2
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "binary image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("binarize - gaussian blur & otsu", result)
    
    # 均值迁移模糊去噪声与二值化
    dst3 = threshold_3(src)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst3 = cv.cvtColor(dst3, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst3
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "binary image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("binarize - mean shift filtering & otsu", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def threshold_1(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return dst

def threshold_2(src):
    blurred = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    _, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return dst

def threshold_3(src):
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    _, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return dst

if __name__ == "__main__":
    OpencvDemo045()

# end of file
