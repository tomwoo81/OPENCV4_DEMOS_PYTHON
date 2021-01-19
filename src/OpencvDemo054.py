#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—对轮廓圆与椭圆拟合
def OpencvDemo054():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/stuff.jpg")
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

    for contour in contours:
        # 椭圆拟合
        (cx, cy), (a, b), angle = cv.fitEllipse(contour)
        
        # 绘制椭圆
        cv.ellipse(dst, (np.int32(cx), np.int32(cy)),
                (np.int32(a/2), np.int32(b/2)), angle, 0, 360, (0, 0, 255), 2, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of ellipse fitting", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("ellipse fitting", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo054()

# end of file
