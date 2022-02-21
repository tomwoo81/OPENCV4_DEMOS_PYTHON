#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# GrabCut图像分割—背景替换
def OpencvDemo119():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/master.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]
    
    rect = (53, 12, src.shape[1] - 100, src.shape[0] - 12)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = cv.grabCut(src, None, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)[0]

    mask2 = np.where((mask == cv.GC_FGD) + (mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
	
	# 对掩膜进行高斯模糊
    mask2 = cv.GaussianBlur(mask2, (5, 5), 0)
	
    bgd = cv.imread("images/river.jpg")
    if bgd is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    bgd = cv.resize(bgd, (width, height))
	
    # 虚化背景
    bgd2 = cv.GaussianBlur(bgd, (0, 0), 15)
	
	# 融合前景与虚化背景
    dst = np.empty(src.shape, dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            w1 = mask2[row, col] / 255
            b, g, r = src[row, col]
            b2, g2, r2 = bgd2[row, col]
            b = b * w1 + b2 * (1 - w1)
            g = g * w1 + g2 * (1 - w1)
            r = r * w1 + r2 * (1 - w1)
            dst[row, col] = (b, g, r)

    h, w, ch = src.shape
    result = np.zeros([h, w * 4, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    mask2 = cv.cvtColor(mask2, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
    result[0 : h, w : w * 2, :] = mask2
    result[0 : h, w * 2 : w * 3, :] = bgd
    result[0 : h, w * 3 : w * 4, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "foreground mask", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "background image", (w * 2 + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image after background replacement", (w * 3 + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("GrabCut image segmentation - background replacement", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo119()

# end of file
