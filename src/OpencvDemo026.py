#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 高斯双边模糊
def OpencvDemo026():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/example.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = cv.bilateralFilter(src, 0, 100, 10)
    
    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "bilateral filter image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("bilateral filter", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo026()

# end of file
