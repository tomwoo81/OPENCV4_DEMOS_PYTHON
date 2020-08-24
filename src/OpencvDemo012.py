#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频文件的读写
def OpencvDemo012():
    logging.basicConfig(level=logging.DEBUG)
    
    # capture = cv.VideoCapture(0) 打开摄像头
    capture = cv.VideoCapture("videos/bike.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv.CAP_PROP_FPS)
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    logging.debug("width: %d, height: %d, fps: %d, count: %d", width, height, fps, count)
    writer = cv.VideoWriter("output/bike2.avi", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                         (np.int(width), np.int(height)), True)
    
    while True:
        ret, frame = capture.read()
        if ret:
            cv.imshow("video-input", frame)
            writer.write(frame)
            
            c = cv.waitKey(50)
            if c == 27: # ESC
                break
        else:
            break
    
    capture.release()
    writer.release()
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo012()

# end of file
