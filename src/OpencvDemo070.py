#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 形态学应用—用基本梯度实现轮廓分析
def OpencvDemo070():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/kd02.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 形态学梯度-基本梯度
    basic = cv.morphologyEx(src, cv.MORPH_GRADIENT, se)

    # 转灰度图像
    gray = cv.cvtColor(basic, cv.COLOR_BGR2GRAY)

    # 全局阈值二值化
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (1, 5), (-1, -1))

    # 膨胀
    dilate = cv.morphologyEx(binary, cv.MORPH_DILATE, se)

    # 轮廓分析
    contours, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = cv.contourArea(c)

        if area < 200:
            continue

        if h > (w * 3) or h < 20:
            continue

        cv.rectangle(dst, (x, y, w, h), (0, 0, 255), 1, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[h : h * 2, 0 : w, :] = dilate
    result[h : h * 2, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "binarised image with basic gradients", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "binarised image after dilation", (10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
    cv.imshow("basic gradients and coutour analysis", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo070()

# end of file
