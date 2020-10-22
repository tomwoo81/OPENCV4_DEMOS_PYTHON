#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—Hu矩实现轮廓匹配
def OpencvDemo053():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/abc.png")
    tpl = cv.imread("images/a5.png")
    if (src is None) or (tpl is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    
    # 轮廓提取
    contours = contours_info(src)
    contours_tpl = contours_info(tpl)
    
    dst = np.copy(src)
    dst_tpl = np.copy(tpl)
    
    for i, _ in enumerate(contours_tpl):
        # 绘制模板轮廓
        if i == 0:
            cv.drawContours(dst_tpl, contours_tpl, i, (0, 0, 255), 2, cv.LINE_8)
        else:
            cv.drawContours(dst_tpl, contours_tpl, i, (0, 255, 0), 2, cv.LINE_8)
    
    # 模板轮廓Hu矩计算
    moments_tpl = cv.moments(contours_tpl[0])
    hu_tpl = cv.HuMoments(moments_tpl)
    
    for i, contour in enumerate(contours):
        # 轮廓Hu矩计算
        moments = cv.moments(contour)
        hu = cv.HuMoments(moments)
        
        # 轮廓匹配
        dist = cv.matchShapes(hu, hu_tpl, cv.CONTOURS_MATCH_I1, 0)
        logging.info("index: %d, distance between 2 shapes: %.3f", i, dist)
        
        if dist < 1:
            logging.info("index: %d, MATCHED", i)
            cv.drawContours(dst, contours, i, (0, 0, 255), 2, cv.LINE_8)
        else:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2, cv.LINE_8)
    
    h, w, ch = tpl.shape
    result = np.zeros([h * 2, w, ch], dtype=tpl.dtype)
    result[0 : h, 0 : w, :] = tpl
    result[h : h * 2, 0 : w, :] = dst_tpl
    cv.putText(result, "template image", (10, 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 1)
    cv.putText(result, "template image with external contours", (10, h + 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 1)
    cv.imshow("contours matching - template", result)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 1)
    cv.putText(result, "image with external contours", (10, h + 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 1)
    cv.imshow("contours matching - results", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def contours_info(image):
    # 二值化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return contours

if __name__ == "__main__":
    OpencvDemo053()

# end of file
