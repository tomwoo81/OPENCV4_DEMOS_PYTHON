#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像形态学—黑帽操作
def OpencvDemo068():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/morph3.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化图像
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15), (-1, -1))

    # 黑帽操作
    blackHat = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, se)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = binary
    blackHat = cv.cvtColor(blackHat, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w * 2 : w * 3, :] = blackHat
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "binarised image", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image after black hat", (w * 2 + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("black hat", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo068()

# end of file
