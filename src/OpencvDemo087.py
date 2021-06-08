#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析—基于帧差法实现移动对象分析
def OpencvDemo087():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/bike.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    ret, src = capture.read()
    if not ret:
        logging.error("could not read a frame!")
        return cv.Error.StsError
    
    prev_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    prev_gray = cv.GaussianBlur(prev_gray, (0, 0), 15)

    winName = "object tracking by frame-difference method"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)

        dst = np.copy(src)

        next_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        next_gray = cv.GaussianBlur(next_gray, (0, 0), 15)

        diff = cv.subtract(next_gray, prev_gray, dtype=cv.CV_16S)
        diff = cv.convertScaleAbs(diff)

        _, binary = cv.threshold(diff, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        k = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(dst, contours, -1, (0, 0, 255), 1, cv.LINE_8); # draw all contours found

        prev_gray = next_gray

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 3, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        binary = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
        result[0 : h, w : w * 2, :] = binary
        result[0 : h, w * 2 : w * 3, :] = dst
        cv.putText(result, "original frame", (10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        cv.putText(result, "moving objects", (w + 10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        cv.putText(result, "processed frame", (w * 2 + 10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        cv.imshow(winName, result)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo087()

# end of file
