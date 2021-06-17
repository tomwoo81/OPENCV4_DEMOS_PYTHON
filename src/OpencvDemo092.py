#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 对象检测—HAAR特征介绍
def OpencvDemo092():
    logging.basicConfig(level=logging.DEBUG)

    # 打开摄像头
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        logging.error("could not open a camera!")
        return cv.Error.StsError
    
    face_detector = cv.CascadeClassifier("models/haarcascade_frontalface_alt_tree.xml")
    smile_detector = cv.CascadeClassifier("models/haarcascade_smile.xml")

    winName = "HAAR cascade classifiers"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    result = None

    while True:
        ret, src = capture.read()
        if not ret:
            break

        c = cv.waitKey(5)

        dst = np.copy(src)

        faces = face_detector.detectMultiScale(src, scaleFactor=1.01, minNeighbors=1,
                                               minSize=(30, 30), maxSize=(400, 400))
        
        for face in faces:
            cv.rectangle(dst, face, (0, 0, 255), 2, cv.LINE_8, 0)

            x, y, w, h = face
            roi = src[y : y + h, x : x + w]

            smiles = smile_detector.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=5,
                                                     minSize=(15, 15), maxSize=(100, 100))
            
            for smile in smiles:
                cv.rectangle(dst[y : y + h, x : x + w], smile, (0, 255, 0), 1)
        
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
    OpencvDemo092()

# end of file
