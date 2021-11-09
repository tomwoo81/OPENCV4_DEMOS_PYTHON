#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

# 基于描述子匹配的已知对象定位
def OpencvDemo097():
    logging.basicConfig(level=logging.DEBUG)
    
    box = cv.imread("images/box.png")
    box_in_scene = cv.imread("images/box_in_scene.png")
    if (box is None) or (box_in_scene is None):
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    cv.imshow("box", box)
    cv.imshow("box in scene", box_in_scene)

    # 检测ORB关键点和提取描述子
    orb = cv.ORB_create()
    kps_box, descs_box = orb.detectAndCompute(box, None)
    kps_bis, descs_bis = orb.detectAndCompute(box_in_scene, None)

    # 描述子匹配 — 暴力匹配
    bf = cv.BFMatcher_create(cv.NORM_HAMMING, True)

    matches = bf.match(descs_box, descs_bis)

    # 筛选较佳匹配点对
    goodMatches = sorted(matches, key = lambda x: x.distance)[:16]

    # 绘制匹配点对
    img_matches = cv.drawMatches(box, kps_box, box_in_scene, kps_bis, goodMatches, None)

    # Localize the object
    pts_box = list()
    pts_bis = list()

    for goodMatch in goodMatches:
        # Get the keypoints from the good matches
        pts_box.append(kps_box[goodMatch.queryIdx].pt)
        pts_bis.append(kps_bis[goodMatch.trainIdx].pt)

    # H, _ = cv.findHomography(np.array(pts_box), np.array(pts_bis), cv.RANSAC) # 有时无法配准
    H, _ = cv.findHomography(np.array(pts_box), np.array(pts_bis), cv.RHO)

    # Get the corners from the image 1 (the object to be detected)
    corners_box = list()
    corners_box.append((0, 0))
    corners_box.append((box.shape[1], 0))
    corners_box.append((box.shape[1], box.shape[0]))
    corners_box.append((0, box.shape[0]))

    corners_bis = cv.perspectiveTransform(np.array(corners_box, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)

    # Draw lines between the corners (the mapped object in the scene - image 2)
    for i in range(4):
        cv.line(img_matches, (np.int(corners_bis[i][0]) + box.shape[1], np.int(corners_bis[i][1]) + 0), 
                             (np.int(corners_bis[(i + 1) % 4][0]) + box.shape[1], np.int(corners_bis[(i + 1) % 4][1]) + 0), (0, 255, 0), 4)

    # Show detected matches and detected object
    cv.imshow("good matches & object detection", img_matches)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo097()

# end of file
