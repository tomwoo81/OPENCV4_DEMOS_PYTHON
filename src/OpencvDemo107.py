#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# BRISK特征提取与描述子匹配
def OpencvDemo107():
    logging.basicConfig(level=logging.DEBUG)
    
    box = cv.imread("images/box.png")
    box_in_scene = cv.imread("images/box_in_scene.png")
    if (box is None) or (box_in_scene is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("box", box)
    cv.imshow("box in scene", box_in_scene)

    # 检测BRISK关键点和提取描述子
    brisk = cv.BRISK_create()
    kps_box, descs_box = brisk.detectAndCompute(box, None)
    kps_bis, descs_bis = brisk.detectAndCompute(box_in_scene, None)

    # 描述子匹配 — 暴力匹配
    bf = cv.BFMatcher_create(cv.NORM_HAMMING, True)

    matches = bf.match(descs_box, descs_bis)

    # 绘制匹配点对
    img_matches = cv.drawMatches(box, kps_box, box_in_scene, kps_bis, matches, None)
    cv.imshow("BRISK descriptors matching", img_matches)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo107()

# end of file
