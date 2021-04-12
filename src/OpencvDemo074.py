#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—提取最大轮廓与编码关键点
def OpencvDemo074():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/case6.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 闭操作
    close = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)

    # 轮廓发现/轮廓分析
    contours, _ = cv.findContours(close, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    dst1 = np.copy(src)

    height, width = src.shape[:2]
    max = 0
    index = -1

    for c in range(len(contours)):
        _, _, w, h = cv.boundingRect(contours[c])
        if h >= height or w >= width:
            continue

        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    
    # 绘制轮廓及其关键点
    dst2 = np.zeros(src.shape, dtype=src.dtype)

    cv.drawContours(dst1, contours, index, (0, 0, 255), 1, cv.LINE_8)
    cv.drawContours(dst2, contours, index, (0, 0, 255), 1, cv.LINE_8)

    pts = cv.approxPolyDP(contours[index], 4, True)

    for pt in pts:
        pt = pt[0]

        cv.circle(dst1, (pt[0], pt[1]), 2, (0, 255, 0), cv.FILLED, cv.LINE_8, 0)
        cv.circle(dst2, (pt[0], pt[1]), 2, (0, 255, 0), cv.FILLED, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    result[h : h * 2, 0 : w, :] = dst1
    result[h : h * 2, w : w * 2, :] = dst2
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "binarised image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "image 1 with results of contour analysis", (10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "image 2 with results of contour analysis", (w + 10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    windowTitle = "retrieval of the maximum contour and its key points"
    cv.namedWindow(windowTitle, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowTitle, (w, h))
    cv.imshow(windowTitle, result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo074()

# end of file
