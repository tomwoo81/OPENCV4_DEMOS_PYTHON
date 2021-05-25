#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 识别与跟踪视频中的特定颜色对象
def OpencvDemo078():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/color_object.mp4")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    num_of_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    logging.info("frame width: {:d}, frame height: {:d}, FPS: {:.0f}, number of frames: {:d}"
                    .format(width, height, fps, num_of_frames))
    
    winName = "object recognition & tracking in video"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)
        
        dst = process_frame(src)

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

def process_frame(src):
    dst = np.copy(src)

    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (15, 15), (-1, -1))

    # 颜色提取 - 红色
    mask = cv.inRange(hsv, (0, 43, 46), (10, 255, 255))

    # 开操作
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se)

    # 发现轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 寻找最大轮廓
    index = -1
    max = 0
    for c in range(len(contours)):
        area = cv.contourArea(contours[c])
        if area > max:
            max = area
            index = c
    
    # 寻找最大轮廓的最小外接矩形
    if index >= 0:
        rotRect = cv.minAreaRect(contours[index])

        cv.ellipse(dst, rotRect, (0, 255, 0), 2, cv.LINE_8)
        cx, cy = rotRect[0]
        cv.drawMarker(dst, (int(cx), int(cy)), (255, 0, 0), cv.MARKER_CROSS, 20, 2, cv.LINE_8)
    
    return dst

if __name__ == "__main__":
    OpencvDemo078()

# end of file
