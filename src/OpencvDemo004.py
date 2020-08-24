#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 图像像素的读写操作
def OpencvDemo004():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    h, w, ch = src.shape
    logging.debug("h: %u, w: %u, ch: %u", h, w, ch)
    for row in range(h):
        for col in range(w):
            b, g, r = src[row, col]
            b = 255 - b
            g = 255 - g
            r = 255 - r
            src[row, col] = [b, g, r]
    cv.imshow("output", src)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo004()

# end of file
