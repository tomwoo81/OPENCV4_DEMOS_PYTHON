#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 均值迁移模糊
def OpencvDemo027():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/example.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = cv.pyrMeanShiftFiltering(src, 15, 30, termcrit=(cv.TERM_CRITERIA_MAX_ITER+cv.TERM_CRITERIA_EPS, 5, 1))
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
    cv.putText(result, "mean shift filtering image", (w + 10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1)
    cv.imshow("mean shift filtering", result)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo027()

# end of file
