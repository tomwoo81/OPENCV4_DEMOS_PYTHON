#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像梯度–拉普拉斯算子
def OpencvDemo033():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/yuan_test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    src = cv.GaussianBlur(src, (0, 0), 1)
    dst = cv.Laplacian(src, cv.CV_32F, ksize=3, delta=128.0) # 八邻域算子
    dst = cv.convertScaleAbs(dst)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "gradient image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("laplacian", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo033()

# end of file
