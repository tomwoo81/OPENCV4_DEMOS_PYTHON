#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# HOG特征描述子—多尺度检测
def OpencvDemo101():
    logging.basicConfig(level=logging.DEBUG)

    src = cv.imread("images/pedestrian_02.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError

    hog = cv.HOGDescriptor()

    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    # Detect pedestrians in the image
    rects, weights = hog.detectMultiScale(src,
                                          winStride = (4, 4),
                                          padding = (8, 8),
                                        #   scale = 1.05,
                                          scale = 1.25,
                                          useMeanshiftGrouping = False)

    dst = np.copy(src)

    for rect in rects:
        cv.rectangle(dst, rect, (0, 255, 0), 2)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.putText(result, "image with results of HOG pedestrian detection", (w + 10, 30), cv.FONT_ITALIC, .8, (0, 0, 255), 1)
    cv.imshow("HOG pedestrian detection", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo101()

# end of file
