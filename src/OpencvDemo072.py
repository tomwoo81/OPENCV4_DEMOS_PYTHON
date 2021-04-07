#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—缺陷检测一
def OpencvDemo072():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/ce_02.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 开操作
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, se)

    # 轮廓发现/轮廓分析
    contours, _ = cv.findContours(open, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    height = src.shape[0]

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = cv.contourArea(c)

        if h > (height // 2):
            continue

        if area < 150:
            continue

        cv.rectangle(dst, (x, y, w, h), (0, 0, 255), 1, cv.LINE_8, 0)
        cv.drawContours(dst, [c], 0, (0, 255, 0), 2, cv.LINE_8)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of contour analysis", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("defect detection - 1", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo072()

# end of file
