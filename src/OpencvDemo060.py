#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—霍夫直线检测二
def OpencvDemo060():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/morph01.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    binary = canny_demo(src)
    
    # 概率霍夫直线检测
    lines = cv.HoughLinesP(binary, 1, np.pi / 180, 50, None, 50, 10)

    dst = np.zeros(src.shape, src.dtype)

    for i in range(len(lines)):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])
        cv.line(dst, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results of linear detection", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("linear detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo060()

# end of file
