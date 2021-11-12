#!/usr/bin/env python3
#coding = utf-8

import logging
import cv2 as cv

# SIFT特征提取—描述子生成
def OpencvDemo099():
    logging.basicConfig(level=logging.DEBUG)
    
    box = cv.imread("images/box.png")
    box_in_scene = cv.imread("images/box_in_scene.png")
    if (box is None) or (box_in_scene is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("box", box)
    cv.imshow("box in scene", box_in_scene)

    # 检测SIFT关键点和提取描述子
    sift = cv.SIFT().create()
    kps_box, descs_box = sift.detectAndCompute(box, None)
    kps_bis, descs_bis = sift.detectAndCompute(box_in_scene, None)

    # 描述子匹配 — 暴力匹配
    bf = cv.BFMatcher_create(cv.NORM_L2, True)

    matches = bf.match(descs_box, descs_bis)

    # 筛选较佳匹配点对
    goodMatches = sorted(matches, key = lambda x: x.distance)[:15]

    # 绘制匹配点对
    img_matches = cv.drawMatches(box, kps_box, box_in_scene, kps_bis, goodMatches, None)
    cv.imshow("SIFT descriptors matching", img_matches)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo099()

# end of file
