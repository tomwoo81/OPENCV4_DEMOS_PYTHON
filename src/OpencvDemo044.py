#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 自适应阈值算法
def OpencvDemo044():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/text1.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    # OTSU二值寻找算法
    t, dst1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    logging.debug("threshold - otsu: {:.0f}".format(t))
    
    # 自适应阈值算法(Gaussian)
    dst2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    dst1 = cv.cvtColor(dst1, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = dst1
    dst2 = cv.cvtColor(dst2, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w * 2 : w * 3, :] = dst2
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 2)
    cv.putText(result, "binary image (otsu)", (w + 10, 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 2)
    cv.putText(result, "binary image (adaptive)", (w * 2 + 10, 30), cv.FONT_ITALIC, 0.8, (0, 0, 255), 2)
    cv.imshow("binarize - adaptive threshold (gaussian)", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo044()

# end of file
