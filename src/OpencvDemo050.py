#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—轮廓面积与弧长
def OpencvDemo050():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/zhifang_ball.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    binary = canny_demo(src)
    
    # 形态学操作-膨胀
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
    
    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    dst1 = np.copy(src)
    dst2 = np.copy(src)
    
    for contour in contours:
        area = cv.contourArea(contour)
        arcLen = cv.arcLength(contour, True)
        if area < 100 or arcLen < 100:
            continue
        
        # 最大外接矩形
        x, y, w, h = cv.boundingRect(contour)
        
        # 绘制矩形
        cv.rectangle(dst1, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_8, 0)
        
        # 最小外接矩形
        rotRect = cv.minAreaRect(contour)
        box = cv.boxPoints(rotRect)
        box = np.int0(box)
        
        # 绘制旋转矩形与中心位置
        cv.drawContours(dst2, [box], 0, (0, 0, 255), 2)
        cx, cy = rotRect[0]
        cv.circle(dst2, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), cv.FILLED, cv.LINE_8, 0)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst1
    result[0 : h, w * 2 : w * 3, :] = dst2
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .7, (0, 0, 255), 1)
    cv.putText(result, "image with up-right bounding rectangles", (w + 10, 30), cv.FONT_ITALIC, .7, (0, 0, 255), 1)
    cv.putText(result, "image with rotated rectangles of the minimum areas", (w * 2 + 10, 30), cv.FONT_ITALIC, .7, (0, 0, 255), 1)
    cv.imshow("bounding rectangles of contours (area >= 100 and arcLen >= 100)", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def canny_demo(src):
    # Canny边缘检测器
    t = 80
    dst = cv.Canny(src, t, t * 2)
    return dst

if __name__ == "__main__":
    OpencvDemo050()

# end of file
