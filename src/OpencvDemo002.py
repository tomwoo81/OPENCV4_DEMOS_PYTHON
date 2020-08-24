#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像色彩空间转换
def OpencvDemo002():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite("output/gray.png", gray)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo002()

# end of file
