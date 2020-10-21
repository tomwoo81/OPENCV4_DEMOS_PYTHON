#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—用几何矩计算轮廓中心与横纵比过滤
def OpencvDemo052():
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
    
    for i, contour in enumerate(contours):
        # 最小外接矩形
        rotRect = cv.minAreaRect(contour)
        w, h = rotRect[1]
        
        # 计算横纵比
        ratio = np.minimum(w, h) / np.maximum(w, h)
        logging.info("index: %d, ratio of the short side to the long side: %.3f", i, ratio)
        
        # 计算轮廓中心
        moments = cv.moments(contour)
        m00 = moments['m00']
        m10 = moments['m10']
        m01 = moments['m01']
        cx = m10 / m00
        cy = m01 / m00
        
        # 绘制轮廓
        if ratio > 0.9:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2, cv.LINE_8)
        elif ratio < 0.5:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2, cv.LINE_8)
        
        # 绘制轮廓中心
        cv.circle(dst, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), cv.FILLED, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of moments calculation", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("moments", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo052()

# end of file
