#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—凸包检测
def OpencvDemo055():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/hand.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # 删除干扰块（开运算）
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)

    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    for contour in contours:
        isHull = cv.isContourConvex(contour)
        logging.info("A convex contour? {}".format('Y' if isHull else 'N'))
        
        hull = cv.convexHull(contour)

        size = len(hull)
        for i in range(size):
            x1, y1 = hull[i][0]
            x2, y2 = hull[(i + 1) % size][0]
            cv.circle(dst, (x1, y1), 4, (255, 0, 0), 2, cv.LINE_8, 0)
            cv.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results of convex hull detection", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("convex hull detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo055()

# end of file
