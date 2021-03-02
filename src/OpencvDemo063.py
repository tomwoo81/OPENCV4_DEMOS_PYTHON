#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像形态学—膨胀与腐蚀二
def OpencvDemo063():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/coins.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 定义结构元素 3x3大小矩形
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))

    # 膨胀
    dilate = cv.dilate(binary, se, None, (-1, -1), 1)

    # 腐蚀
    erode = cv.erode(binary, se, None, (-1, -1), 1)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[h : h * 2, 0 : w, :] = dilate
    erode = cv.cvtColor(erode, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[h : h * 2, w : w * 2, :] = erode
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "binarised image", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image after dilation", (10, h + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image after erosion", (w + 10, h + 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("dilation & erosion", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo063()

# end of file
