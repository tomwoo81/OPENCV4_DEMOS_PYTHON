#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像形态学—图像梯度
def OpencvDemo069():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/wm.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 基本梯度
    basic = cv.morphologyEx(src, cv.MORPH_GRADIENT, se)

    # 外梯度
    dilate = cv.morphologyEx(src, cv.MORPH_DILATE, se)
    external = cv.subtract(dilate, src)

    # 内梯度
    erode = cv.morphologyEx(src, cv.MORPH_ERODE, se)
    internal = cv.subtract(src, erode)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = basic
    result[h : h * 2, 0 : w, :] = external
    result[h : h * 2, w : w * 2, :] = internal
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with basic gradients", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with external gradients", (10, h + 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with internal gradients", (w + 10, h + 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("gradients", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo069()

# end of file
