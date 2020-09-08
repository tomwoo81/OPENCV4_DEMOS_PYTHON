#!/usr/bin/env python3
#coding = utf-8

from sys import argv
from OpencvDemo001 import OpencvDemo001
from OpencvDemo002 import OpencvDemo002
from OpencvDemo003 import OpencvDemo003
from OpencvDemo004 import OpencvDemo004
from OpencvDemo005 import OpencvDemo005
from OpencvDemo006 import OpencvDemo006
from OpencvDemo007 import OpencvDemo007
from OpencvDemo008 import OpencvDemo008
from OpencvDemo009 import OpencvDemo009
from OpencvDemo010 import OpencvDemo010
from OpencvDemo011 import OpencvDemo011
from OpencvDemo012 import OpencvDemo012
from OpencvDemo013 import OpencvDemo013
from OpencvDemo014 import OpencvDemo014
from OpencvDemo015 import OpencvDemo015
from OpencvDemo016 import OpencvDemo016
from OpencvDemo017 import OpencvDemo017
from OpencvDemo018 import OpencvDemo018
from OpencvDemo019 import OpencvDemo019
from OpencvDemo020 import OpencvDemo020
from OpencvDemo021 import OpencvDemo021
from OpencvDemo022 import OpencvDemo022
from OpencvDemo023 import OpencvDemo023
from OpencvDemo024 import OpencvDemo024
from OpencvDemo025 import OpencvDemo025
from OpencvDemo026 import OpencvDemo026
from OpencvDemo027 import OpencvDemo027
from OpencvDemo028 import OpencvDemo028
from OpencvDemo029 import OpencvDemo029
from OpencvDemo030 import OpencvDemo030

def main(argv):
    argc = len(argv)
    
    if argc < 2:
        print("No arguments!")
        return -1
    
    try:
        demoId = int(argv[1])
    except:
        print("The argument is invalid!")
    
    if demoId == 1:
        # 图像读取与显示
        ret = OpencvDemo001()
    elif demoId == 2:
        # 图像色彩空间转换
        ret = OpencvDemo002()
    elif demoId == 3:
        # 图像对象的创建与赋值
        ret = OpencvDemo003()
    elif demoId == 4:
        # 图像像素的读写操作
        ret = OpencvDemo004()
    elif demoId == 5:
        # 图像像素的算术操作
        ret = OpencvDemo005()
    elif demoId == 6:
        # LUT的作用与用法
        ret = OpencvDemo006()
    elif demoId == 7:
        # 图像像素的逻辑操作
        ret = OpencvDemo007()
    elif demoId == 8:
        # 通道分离与合并
        ret = OpencvDemo008()
    elif demoId == 9:
        # 图像色彩空间转换
        ret = OpencvDemo009()
    elif demoId == 10:
        # 图像像素值统计
        ret = OpencvDemo010()
    elif demoId == 11:
        # 像素归一化
        ret = OpencvDemo011()
    elif demoId == 12:
        # 视频文件的读写
        ret = OpencvDemo012()
    elif demoId == 13:
        # 图像翻转
        ret = OpencvDemo013()
    elif demoId == 14:
        # 图像插值
        ret = OpencvDemo014()
    elif demoId == 15:
        # 几何形状绘制
        ret = OpencvDemo015()
    elif demoId == 16:
        # 图像ROI与ROI操作
        ret = OpencvDemo016()
    elif demoId == 17:
        # 图像直方图
        ret = OpencvDemo017()
    elif demoId == 18:
        # 图像直方图均衡化
        ret = OpencvDemo018()
    elif demoId == 19:
        # 图像直方图比较
        ret = OpencvDemo019()
    elif demoId == 20:
        # 图像直方图反向投影
        ret = OpencvDemo020()
    elif demoId == 21:
        # 图像卷积操作
        ret = OpencvDemo021()
    elif demoId == 22:
        # 图像均值与高斯模糊
        ret = OpencvDemo022()
    elif demoId == 23:
        # 中值模糊
        ret = OpencvDemo023()
    elif demoId == 24:
        # 图像噪声
        ret = OpencvDemo024()
    elif demoId == 25:
        # 图像去噪声
        ret = OpencvDemo025()
    elif demoId == 26:
        # 高斯双边模糊
        ret = OpencvDemo026()
    elif demoId == 27:
        # 均值迁移模糊
        ret = OpencvDemo027()
    elif demoId == 28:
        # 图像积分图算法
        ret = OpencvDemo028()
    elif demoId == 29:
        # 快速的图像边缘滤波算法
        ret = OpencvDemo029()
    elif demoId == 30:
        # OpenCV自定义的滤波器
        ret = OpencvDemo030()
    else:
        print("The argument is invalid!")
        return -1
    
    return ret

if __name__ == "__main__":
    main(argv)

# end of file
