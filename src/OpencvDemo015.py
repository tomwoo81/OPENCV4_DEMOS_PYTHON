#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 几何形状绘制
def OpencvDemo015():
    logging.basicConfig(level=logging.DEBUG)
    
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv.rectangle(image, (100, 100), (300, 300), (255, 0, 0), 2, cv.LINE_8, 0)
    cv.circle(image, (256, 256), 50, (0, 0, 255), 2, cv.LINE_8, 0)
    cv.ellipse(image, (256, 256), (150, 50), 360, 0, 360, (0, 255, 0), 2, cv.LINE_8, 0)
    cv.imshow("image", image)
    
    cv.waitKey(0)
    
    for i in range(100000):
        image[:,:,:] = 0
        
        x1 = np.random.rand() * 512
        y1 = np.random.rand() * 512
        x2 = np.random.rand() * 512
        y2 = np.random.rand() * 512
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        
        cv.line(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 4, cv.LINE_8, 0)
        
#         cv.rectangle(image, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (b, g, r), 1, cv.LINE_8, 0)
        
        cv.imshow("image", image)
        
        c = cv.waitKey(20)
        if c == 27: # ESC
            break
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo015()

# end of file
