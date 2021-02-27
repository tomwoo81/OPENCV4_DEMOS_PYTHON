#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—霍夫圆检测
def OpencvDemo061():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test_coins.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 2, 2)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 2, 10, None, 100, 80, 20, 100)

    dst = np.copy(src)

    for c in circles[0, :]:
        cx, cy, r = c

        # 绘制圆
        cv.circle(dst, (int(cx), int(cy)), 2, (0, 255, 0), cv.FILLED, cv.LINE_8, 0)
        cv.circle(dst, (int(cx), int(cy)), int(r), (0, 0, 255), 2, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.putText(result, "image with results of circular detection", (10, h + 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.imshow("circular detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo061()

# end of file
