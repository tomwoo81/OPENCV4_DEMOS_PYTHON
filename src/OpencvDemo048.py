#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—轮廓发现
def OpencvDemo048():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/yuan_test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    binary = canny_demo(src)
    
    # 轮廓发现与绘制
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dst1 = np.copy(src)
    cv.drawContours(dst1, contours, -1, (0, 0, 255), 2, cv.LINE_8) # draw all contours found
    
    binary = threshold_demo(src)
    
    # 轮廓发现与绘制
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dst2 = np.copy(src)
    cv.drawContours(dst2, contours, -1, (0, 0, 255), 2, cv.LINE_8) # draw all contours found
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst1
    result[0 : h, w * 2 : w * 3, :] = dst2
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 2)
    cv.putText(result, "image with all contours (canny)", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 2)
    cv.putText(result, "image with all contours (threshold)", (w * 2 + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 2)
    cv.imshow("contours finding", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 100
    dst = cv.Canny(src, t, t * 2)
    return dst

def threshold_demo(src):
    # 去噪声与二值化
    blurred = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    _, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return dst

if __name__ == "__main__":
    OpencvDemo048()

# end of file
