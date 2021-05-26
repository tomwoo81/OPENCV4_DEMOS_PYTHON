#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析—背景消除与前景ROI提取
def OpencvDemo080():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/vtest.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    num_of_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    logging.info("frame width: {:d}, frame height: {:d}, FPS: {:.0f}, number of frames: {:d}"
                    .format(width, height, fps, num_of_frames))
    
    mog2 = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
    winName = "background elimination & foreground extraction"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)
        
        dst = process_frame(mog2, src)

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

def process_frame(bgSub, src):
    dst = np.copy(src)

    # 提取前景，生成mask
    mask = bgSub.apply(src)

    # 定义结构元素
    se = cv.getStructuringElement(cv.MORPH_RECT, (1, 5), (-1, -1))

    # 开操作
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se)

    # 发现轮廓
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in range(len(contours)):
        # 过滤面积较小的轮廓
        area = cv.contourArea(contours[c])
        if area < 100:
            continue

        # 寻找轮廓的最小外接矩形
        rotRect = cv.minAreaRect(contours[c])

        cv.ellipse(dst, rotRect, (0, 255, 0), 2, cv.LINE_8)
        cx, cy = rotRect[0]
        cv.drawMarker(dst, (int(cx), int(cy)), (255, 0, 0), cv.MARKER_CROSS, 20, 2, cv.LINE_8)
    
    return dst

if __name__ == "__main__":
    OpencvDemo080()

# end of file
