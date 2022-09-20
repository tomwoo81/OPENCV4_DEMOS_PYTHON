#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 前景背景图像合成
def compose_images():
    logging.basicConfig(level=logging.DEBUG)
    
    # Generate a composite image from foreground and background images
    
    foreground = cv.imread("images/greenback.png")
    if foreground is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    # BGR space -> HSV space
    hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
    # 提取前景与背景区域
    mask_inv = cv.inRange(hsv, (35, 43, 46), (77, 255, 255))
    mask = cv.bitwise_not(mask_inv)
    # 删除干扰区域（开运算）
    k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    # 使前景与背景间边界光滑（高斯模糊）
    mask = cv.GaussianBlur(mask, (9, 9), 0)
    
    background = cv.imread("images/river.jpg")
    if background is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    background = cv.resize(background, (foreground.shape[1], foreground.shape[0]))
    
    mask_float = cv.normalize(mask.astype(np.float32), None, 1.0, norm_type=cv.NORM_INF)
    mask_inv_float = cv.subtract(1, mask_float)
    mask_float = cv.cvtColor(mask_float, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    mask_inv_float = cv.cvtColor(mask_inv_float, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    foreground_masked = cv.multiply(foreground.astype(np.float32), mask_float)
    background_masked = cv.multiply(background.astype(np.float32), mask_inv_float)
    composite = cv.add(foreground_masked, background_masked).astype(np.uint8)
    
    h, w, ch = foreground.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=foreground.dtype)
    result[0 : h, 0 : w, :] = foreground
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = mask
    result[h : h * 2, 0 : w, :] = background
    result[h : h * 2, w : w * 2, :] = composite
    cv.putText(result, "foreground", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "mask", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "background", (10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    cv.putText(result, "composite", (w + 10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    windowTitle = "image composition"
    cv.namedWindow(windowTitle, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowTitle, (w, h))
    cv.imshow(windowTitle, result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    compose_images()

# end of file
