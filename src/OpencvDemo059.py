#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—霍夫直线检测
def OpencvDemo059():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/sudoku.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    binary = canny_demo(src)
    
    # 标准霍夫直线检测
    lines = cv.HoughLines(binary, 1, np.pi / 180, 150, None, 0, 0)

    dst = np.copy(src)
    
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta); b = np.sin(theta)
        x0 = rho * a; y0 = rho * b
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(dst, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of linear detection", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("linear detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo059()

# end of file
