#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析—基于连续自适应均值迁移的对象移动分析
def OpencvDemo089():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/balltest.mp4")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    # 读取第一帧
    ret, src = capture.read()
    if not ret:
        logging.error("could not read a frame!")
        return cv.Error.StsError
    
    x, y, w, h = cv.selectROI("select a region of interest", src, False, False)
    tracking_window = (x, y, w, h)

    # 获取ROI直方图
    src_roi = src[y : y + h, x : x + w]
    hsv_roi = cv.cvtColor(src_roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, (26, 43, 46), (34, 255, 255))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    winName = "object tracking by continously adaptive mean shift method"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)

        dst = np.copy(src)

        # 图像直方图反向投影
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        back_proj = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 通过均值迁移方法搜索更新ROI区域
        term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1)
        tracking_box, tracking_window = cv.CamShift(back_proj, tracking_window, term_crit)

        # 绘制窗口
        cv.rectangle(dst, tracking_window, (0, 0, 255), 2)

        # 绘制ROI区域
        cv.ellipse(dst, tracking_box, (0, 255, 0), 2)

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 3, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        back_proj = cv.cvtColor(back_proj, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
        result[0 : h, w : w * 2, :] = back_proj
        result[0 : h, w * 2 : w * 3, :] = dst
        cv.putText(result, "original frame", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.putText(result, "back projection", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.putText(result, "processed frame", (w * 2 + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 1)
        cv.imshow(winName, result)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo089()

# end of file
