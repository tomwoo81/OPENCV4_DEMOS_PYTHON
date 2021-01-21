#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—点多边形测试
def OpencvDemo057():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/my_mask.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dst = np.zeros(src.shape, dtype=np.float32)
    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            dist = cv.pointPolygonTest(contours[0], (col, row), True)

            if dist == 0:
                dst[row, col] = (255, 255, 255)
            if dist > 0:
                dst[row, col] = (255 - dist, 0, 0)
            if dist < 0:
                dst[row, col] = (0, 0, 255 + dist)
    
    dst = cv.convertScaleAbs(dst)
    dst = np.uint8(dst)

    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 0.8, (0, 255, 0), 1)
    cv.putText(result, "image with results of point-in-contour testing", (10, h + 30), cv.FONT_ITALIC, 0.8, (0, 255, 0), 1)
    cv.imshow("point-in-contour testing", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo057()

# end of file
