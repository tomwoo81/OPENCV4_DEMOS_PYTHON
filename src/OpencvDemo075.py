#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 图像去水印/修复
def OpencvDemo075():
    logging.basicConfig(level=logging.DEBUG)
    
    # inpainting - 1
    src = cv.imread("images/wm.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # inpainting mask
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (100, 43, 46), (124, 255, 255))
    se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
    mask = cv.dilate(mask, se)

    # inpainting (image restoration)
    dst = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = mask
    result[0 : h, w * 2 : w * 3, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "inpainting mask", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "inpainted image", (w * 2 + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("inpainting - 1", result)

    # inpainting - 2
    src = cv.imread("images/master2.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # inpainting mask
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (100, 43, 46), (124, 255, 255))
    se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))
    mask = cv.dilate(mask, se)

    # inpainting (image restoration)
    dst = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)

    h, w, ch = src.shape
    result = np.zeros([h, w * 3, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = mask
    result[0 : h, w * 2 : w * 3, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "inpainting mask", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "inpainted image", (w * 2 + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("inpainting - 2", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo075()

# end of file
