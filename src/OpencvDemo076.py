#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像透视变换应用
def OpencvDemo076():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/st_02.png")
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

    # 寻找最大轮廓
    contours, _ = cv.findContours(open, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max = 0
    index = -1

    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    
    # 寻找最小外接矩形
    rect = cv.minAreaRect(contours[index])
    height, width = np.int0(rect[1])
    logging.debug("width: {:d}, height: {:d}, angle: {:.3f}".format(width, height, rect[2]))

    vertices = cv.boxPoints(rect)
    src_pts = np.int0(vertices)
    dst1 = np.copy(src)

    for i in range(src_pts.shape[0]):
        logging.debug("x: {:d}, y: {:d}".format(src_pts[i][0], src_pts[i][1]))
        cv.drawMarker(dst1, (src_pts[i][0], src_pts[i][1]), (0, 0, 255), cv.MARKER_CROSS, 20, 1, cv.LINE_8)
    
    dst_pts = list()
    dst_pts.append((width, height))
    dst_pts.append((0, height))
    dst_pts.append((0, 0))
    dst_pts.append((width, 0))

    # 透视变换
    M, _ = cv.findHomography(src_pts, np.array(dst_pts))
    dst2 = cv.warpPerspective(src, M, (width, height))

    if height < width:
        dst2 = cv.rotate(dst2, cv.ROTATE_90_CLOCKWISE)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst1
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with source points", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("perspective transformation - A", result)
    cv.putText(dst2, "image after perspective transformation", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("perspective transformation - B", dst2)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo076()

# end of file
