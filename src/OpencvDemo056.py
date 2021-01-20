#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—直线拟合与极值点寻找
def OpencvDemo056():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/twolines.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    binary = canny_demo(src)

    # 形态学操作-膨胀
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dst = np.copy(src)

    # 直线拟合与极值点寻找
    for contour in contours:
        # 最大外接矩形
        _, _, w, h = cv.boundingRect(contour)

        rect_max = max(w, h)
        if (rect_max < 30) or (rect_max > 300):
            continue

        # 直线拟合
        vx, vy, x0, y0 = cv.fitLine(contour, cv.DIST_L1, 0, 0.01, 0.01)

        # 直线参数斜率k与截矩b
        k = vy / vx
        b = y0 - k * x0

        # 寻找轮廓极值点
        y_min = 10000
        y_max = 0
        for pt in contour:
            _, py = pt[0]
            if y_min > py:
                y_min = py
            if y_max < py:
                y_max = py
        x_min = (y_min - b) / k
        x_max = (y_max - b) / k

        cv.line(dst, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .4, (0, 0, 255), 1)
    cv.putText(result, "image with results of line fitting", (w + 10, 30), cv.FONT_ITALIC, .4, (0, 0, 255), 1)
    cv.imshow("line fitting", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo056()

# end of file
