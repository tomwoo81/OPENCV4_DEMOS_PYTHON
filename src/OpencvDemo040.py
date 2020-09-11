#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二值图像介绍
def OpencvDemo040():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = np.zeros(gray.shape, dtype=np.uint8)
#     t = 127
    t = cv.mean(gray)[0]
    logging.debug("threshold: {:.3f}".format(t))
    
    # 直接读取图像像素
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            pv = gray[i, j]
            if pv > t:
                dst[i, j] = 255
            else:
                dst[i, j] = 0
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "binary image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("binarize", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo040()

# end of file
