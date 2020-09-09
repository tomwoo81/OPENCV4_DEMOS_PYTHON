#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像锐化
def OpencvDemo034():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test1.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    sharpen_op = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]], dtype=np.float32)
    dst = cv.filter2D(src, cv.CV_32F, sharpen_op)
    dst = cv.convertScaleAbs(dst)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.putText(result, "sharpen image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 1)
    cv.imshow("sharpen", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo034()

# end of file
