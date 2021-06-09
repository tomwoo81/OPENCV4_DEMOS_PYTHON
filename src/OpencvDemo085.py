#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 视频分析-移动对象的KLT光流跟踪算法之二
def OpencvDemo085():
    logging.basicConfig(level=logging.DEBUG)

    capture = cv.VideoCapture("videos/vtest.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    ret, src = capture.read()
    if not ret:
        logging.error("could not read a frame!")
        return cv.Error.StsError
    
    # detector parameters
    featuresParams = dict(maxCorners = 5000, qualityLevel = 0.01, minDistance = 10, blockSize=3)

    # detecting corners
    old_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    old_pts = cv.goodFeaturesToTrack(old_gray, **featuresParams)
    init_pts = old_pts
    logging.info("number of feature points: {:d}".format(len(init_pts)))
    
    winName = "Lucas-Kanade optical flow tracking"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(50)

        dst = np.copy(src)

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        # tracker parameters
        flowParams = dict(winSize=(31, 31), maxLevel=3, criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 30, 0.01))

        # calculating an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
        if len(old_pts) > 0:
            pts, status, _ = cv.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, **flowParams)
        
        k = 0

        for i, (old_pt, pt, init_pt) in enumerate(zip(old_pts, pts, init_pts)):
            x0, y0 = old_pt[0]
            x, y = pt[0]

            # judging the status and the distance
            dist = abs(x - x0) + abs(y - y0)
            if status[i] and dist > 2:
                old_pts[k] = old_pt
                pts[k] = pt
                init_pts[k] = init_pt
                k += 1
                
                # drawing circles around feature points and lines on their optical flows
                xi, yi = np.int32(init_pt[0])
                x, y = np.int32(pt[0])
                b = np.random.randint(0, 256)
                g = np.random.randint(0, 256)
                r = np.random.randint(0, 256)
                cv.circle(dst, (x, y), 3, (int(b), int(g), int(r)), cv.FILLED)
                cv.line(dst, (xi, yi), (x, y), (int(b), int(g), int(r)), 2)
        
        # reserving the valid feature points
        old_pts = old_pts[:k]
        pts = pts[:k]
        init_pts = init_pts[:k]

        # updating old gray image and old feature points
        old_gray = gray
        old_pts = pts
        pts = None

        if len(init_pts) < 40:
            # re-detecting feature points
            feature_pts = cv.goodFeaturesToTrack(gray, **featuresParams)
            old_pts = np.vstack((old_pts, feature_pts))
            init_pts = np.vstack((init_pts, feature_pts))
            logging.info("number of feature points: {:d}".format(len(init_pts)))
        
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
    OpencvDemo085()

# end of file