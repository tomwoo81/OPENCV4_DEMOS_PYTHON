#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像积分图算法
def OpencvDemo028():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test1.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    integral_image = cv.integral(src, sdepth=cv.CV_32S)
    dst = blur_demo(src, integral_image)
    
    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "blur image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("blur using integral image", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def blur_demo(src, integral_image):
    h, w, ch = src.shape
    x1 = 0; y1 = 0
    x2 = 0; y2 = 0
    ksize = 15
    radius = ksize // 2
    dst = np.zeros(src.shape, src.dtype)
    
    for yc in range(0, h, 1):
        y1 = 0 if (yc - radius - 1) < 0 else yc - radius - 1
        y2 = h - 1 if (yc + radius) >= h else yc + radius
        for xc in range(0, w, 1):
            x1 = 0 if (xc - radius - 1) < 0 else xc - radius - 1
            x2 = w - 1 if (xc + radius) >= w else xc + radius
            num = (x2 - x1) * (y2 - y1)
            for i in range(0, ch, 1):
                s = get_block_sum(integral_image, x1, y1, x2, y2, i)
                dst[yc, xc][i] = s // num
    
    return dst

def get_block_sum(integral_image, x1, y1, x2, y2, i):
    tl = integral_image[y1 + 1, x1 + 1][i]
    tr = integral_image[y1 + 1, x2 + 1][i]
    bl = integral_image[y2 + 1, x1 + 1][i]
    br = integral_image[y2 + 1, x2 + 1][i]
    
    s = (br - bl - tr + tl)
    
    return s

if __name__ == "__main__":
    OpencvDemo028()

# end of file
