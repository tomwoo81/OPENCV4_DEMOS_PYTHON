#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析–稠密光流分析
def OpencvDemo086():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/vtest.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    ret, src = capture.read()
    if not ret:
        logging.error("could not read a frame!")
        return cv.Error.StsError
    
    prev_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(src)

    winName = "Gunnar Farneback optical flow tracking"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(20)

        next_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], None, None, True)

        hsv[..., 0] = ang / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        dst = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)

        prev_gray = next_gray

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 2, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        result[0 : h, w : w * 2, :] = dst
        cv.putText(result, "original frame", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.putText(result, "processed frame", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.imshow(winName, result)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo086()

# end of file
