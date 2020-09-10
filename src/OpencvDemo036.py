#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# Canny边缘检测器
def OpencvDemo036():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = cv.Canny(src, 100, 300) # t1 = 100, t2 = 3 * t1 = 300
    dst = cv.bitwise_and(src, src, mask=dst) # make edges colourful
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "edge image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.imshow("canny", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo036()

# end of file
