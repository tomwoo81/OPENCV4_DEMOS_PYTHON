#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频读写与处理
def OpencvDemo077():
    logging.basicConfig(level=logging.DEBUG)

    # 打开摄像头
    # capture = cv.VideoCapture(0)

    # 打开文件
    capture = cv.VideoCapture("videos/roadcars.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    num_of_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    logging.info("frame width: {:d}, frame height: {:d}, FPS: {:.0f}, number of frames: {:d}"
                    .format(width, height, fps, num_of_frames))

    cv.namedWindow("video processing", cv.WINDOW_AUTOSIZE)
    index = 0
    result = None

    while(True):
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)
        if c >= 49: # '1'
            index = c - 49
        
        dst = process_frame(src, index)

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 2, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        if index == 2:
            dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
        result[0 : h, w : w * 2, :] = dst
        cv.putText(result, "original frame", (10, 20), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
        cv.putText(result, "processed frame", (w + 10, 20), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
        cv.imshow("video processing", result)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def process_frame(src, opt):
    if opt == 0:
        dst = cv.bitwise_not(src)
    elif opt == 1:
        dst = cv.GaussianBlur(src, (0, 0), 5)
    elif opt == 2:
        dst = cv.Canny(src, 100, 200)
    else:
        dst = np.copy(src)
    
    return dst

if __name__ == "__main__":
    OpencvDemo077()

# end of file
