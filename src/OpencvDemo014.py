#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像插值
def OpencvDemo014():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    h, w = src.shape[:2]
    logging.debug("height: %d, width: %d", h, w)
    
    dst = cv.resize(src, (w*2, h*2), fx=0.75, fy=0.75, interpolation=cv.INTER_NEAREST)
    cv.imshow("INTER_NEAREST", dst)
    
    dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_LINEAR)
    cv.imshow("INTER_LINEAR", dst)
    
    dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_CUBIC)
    cv.imshow("INTER_CUBIC", dst)
    
    dst = cv.resize(src, (w*2, h*2), interpolation=cv.INTER_LANCZOS4)
    cv.imshow("INTER_LANCZOS4", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo014()

# end of file
