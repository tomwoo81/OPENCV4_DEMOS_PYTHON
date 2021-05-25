#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析—背景/前景提取
def OpencvDemo079():
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
    
    mog2 = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=1000, detectShadows=False)
    winName = "foreground/background segmentation"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)
        
        mask = mog2.apply(src)
        background = mog2.getBackgroundImage()

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 3, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
        result[0 : h, w : w * 2, :] = mask
        result[0 : h, w * 2 : w * 3, :] = background
        cv.putText(result, "original frame", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.putText(result, "foreground mask", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.putText(result, "background image", (w * 2 + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.imshow(winName, result)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo079()

# end of file
