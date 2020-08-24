#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# LUT的作用与用法
def OpencvDemo006():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/test.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", src)
    dst = cv.applyColorMap(src, cv.COLORMAP_COOL)
    cv.imshow("output", dst)
    
    # 伪色彩
    image = cv.imread("images/canjian.jpg")
    color_image = cv.applyColorMap(image, cv.COLORMAP_JET)
    cv.imshow("image", image)
    cv.imshow("color_image", color_image)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo006()

# end of file
