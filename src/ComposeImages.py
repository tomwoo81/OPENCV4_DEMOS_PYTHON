#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 前景背景图像合成
def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # Generate a composite image from foreground and background images
    foreground = cv.imread("images/greenback.png")
    if foreground is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    cv.imshow("foreground", foreground)
    
    hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (35, 43, 46), (77, 255, 255))
    cv.imshow("mask", mask)
    mask_inv = cv.bitwise_not(mask)
    cv.imshow("mask_inv", mask_inv)
    
    foreground_masked = np.zeros(foreground.shape[:2])
    foreground_masked = cv.bitwise_and(foreground, foreground, foreground_masked, mask_inv)
    
    background = cv.imread("images/river.jpg")
    if background is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    background = cv.resize(background, (foreground.shape[1], foreground.shape[0]))
    cv.imshow("background", background)
    
    background_masked = np.zeros(background.shape[:2])
    background_masked = cv.bitwise_and(background, background, background_masked, mask)
    
    composite = cv.bitwise_or(foreground_masked, background_masked)
    cv.imshow("composite", composite)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    main()

# end of file
