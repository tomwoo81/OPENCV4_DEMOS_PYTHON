#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 二维码检测与识别
def OpencvDemo120():
    logging.basicConfig(level=logging.DEBUG)
    
    src = cv.imread("images/qrcode_06.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = np.copy(src)
	
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
	
    detector = cv.QRCodeDetector()
    info, points, straight_qrcode = detector.detectAndDecode(gray)
	
    cv.drawContours(dst, [np.int32(points)], 0, (0, 0, 255), 2)
    points = points[0]
    cv.circle(dst, (points[0][0], points[0][1]), 2, (0, 255, 0), cv.FILLED)
    cv.circle(dst, (points[1][0], points[1][1]), 2, (255, 0, 0), cv.FILLED)
    cv.circle(dst, (points[3][0], points[3][1]), 2, (0, 255, 255), cv.FILLED)

    cv.imshow("rectified and binarized QR code", straight_qrcode)

    h, w, ch = src.shape
    result = np.zeros([h * 2, w, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[h : h * 2, 0 : w, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.putText(result, "image with results of QR code detection and decoding", (10, h + 30), cv.FONT_ITALIC, .5, (0, 0, 255), 1)
    cv.putText(result, info, (10, h + 350), cv.FONT_ITALIC, .4, (255, 0, 0), 1)
    cv.imshow("QR code detection and decoding", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo120()

# end of file
