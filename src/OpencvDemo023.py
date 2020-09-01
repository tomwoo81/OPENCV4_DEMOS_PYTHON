#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# 中值模糊
def OpencvDemo023():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/sp_noise.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    
    dst = cv.medianBlur(src, 5)
    cv.imshow("median blur ksize=5", dst)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo023()

# end of file
