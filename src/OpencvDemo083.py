#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 角点检测–亚像素级别角点检测
def OpencvDemo083():
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
    
    winName = "Shi-Tomasi corner detection"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)
        
        dst = process(src)

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

def process(src):
    dst = np.copy(src)

    # detector parameters
    maxCorners = 100
    qualityLevel = 0.05
    minDistance = 10

    # detecting corners
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    logging.info("number of corners: {:d}".format(len(corners)))

    # drawing circles around corners
    for c in corners:
        x, y = np.int32(c[0])
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        cv.circle(dst, (x, y), 5, (int(b), int(g), int(r)), 2)
    
    # refinement parameters
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 40, 0.001)

    # refining corner locations
    corners = cv.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

    # printing refined corner locations
    for i in range(corners.shape[0]):
        logging.info("refined corner location [{:d}]: ({:.3f}, {:.3f})".format(i, corners[i, 0, 0], corners[i, 0, 1]))
    
    return dst

if __name__ == "__main__":
    OpencvDemo083()

# end of file
