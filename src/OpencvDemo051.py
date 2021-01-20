#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像分析—使用轮廓逼近
def OpencvDemo051():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/contours.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # 二值化
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # 轮廓发现
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    dst = np.copy(src)
    
    for contour in contours:
        # 最小外接矩形
        rotRect = cv.minAreaRect(contour)
        
        # 最小外接矩形的中心位置
        cx, cy = rotRect[0]
        
        # 多边形逼近
        result = cv.approxPolyDP(contour, 4, True)
        numVertices = result.shape[0]
        logging.info("number of vertices: %d", numVertices)
        
        # 形状判断
        if numVertices == 3:
            cv.putText(dst, "triangle", (int(cx), int(cy)), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv.LINE_8)
        elif numVertices == 4:
            cv.putText(dst, "rectangle", (int(cx), int(cy)), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv.LINE_8)
        elif numVertices == 6:
            cv.putText(dst, "polygon", (int(cx), int(cy)), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv.LINE_8)
        elif numVertices > 10:
            cv.putText(dst, "circle", (int(cx), int(cy)), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2, cv.LINE_8)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results of polygonal approximation", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("polygonal approximation", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo051()

# end of file
