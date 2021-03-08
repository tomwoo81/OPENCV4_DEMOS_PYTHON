#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像形态学—开闭操作时候结构元素应用演示
def OpencvDemo066():
    logging.basicConfig(level=logging.DEBUG)
    
    if cv.Error.StsOk != open_demo():
        logging.error("open_demo() failed!")
        return cv.Error.StsError
    
    if cv.Error.StsOk != close_demo():
        logging.error("close_demo() failed!")
        return cv.Error.StsError
    
    return cv.Error.StsOk

def open_demo():
    src = cv.imread("images/fill.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (20, 1), (-1, -1))

    # 开操作
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, se)

    # 提取轮廓
    contours, hierarchy = cv.findContours(open, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        y = y - 10
        h = 12
        cv.rectangle(dst, (x, y, w, h), (0, 0, 255), 1, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    open = cv.cvtColor(open, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[h : h * 2, 0 : w, :] = open
    result[h : h * 2, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "binarised image", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image after opening", (10, h + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, h + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("opening demo", result)

    cv.waitKey(0)
    
    cv.destroyWindow("opening demo")
    
    return cv.Error.StsOk

def close_demo():
    src = cv.imread("images/morph3.png")

    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15), (-1, -1))

    # 闭操作
    close = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    close = cv.cvtColor(close, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w * 2 : w * 3, :] = close
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "binarised image", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image after closing", (w * 2 + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("closing demo", result)

    cv.waitKey(0)
    
    cv.destroyWindow("closing demo")
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo066()

# end of file
