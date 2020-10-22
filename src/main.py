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
from OpencvDemo031 import OpencvDemo031
from OpencvDemo032 import OpencvDemo032
from OpencvDemo033 import OpencvDemo033
from OpencvDemo034 import OpencvDemo034
from OpencvDemo035 import OpencvDemo035
from OpencvDemo036 import OpencvDemo036
from OpencvDemo037 import OpencvDemo037
from OpencvDemo038 import OpencvDemo038
from OpencvDemo039 import OpencvDemo039
from OpencvDemo040 import OpencvDemo040
from OpencvDemo041 import OpencvDemo041
from OpencvDemo042 import OpencvDemo042
from OpencvDemo043 import OpencvDemo043
from OpencvDemo044 import OpencvDemo044
from OpencvDemo045 import OpencvDemo045
from OpencvDemo046 import OpencvDemo046
from OpencvDemo047 import OpencvDemo047
from OpencvDemo048 import OpencvDemo048
from OpencvDemo049 import OpencvDemo049
from OpencvDemo050 import OpencvDemo050
from OpencvDemo051 import OpencvDemo051
from OpencvDemo052 import OpencvDemo052
from OpencvDemo053 import OpencvDemo053
# from OpencvDemo054 import OpencvDemo054
# from OpencvDemo055 import OpencvDemo055
# from OpencvDemo056 import OpencvDemo056
# from OpencvDemo057 import OpencvDemo057
# from OpencvDemo058 import OpencvDemo058
# from OpencvDemo059 import OpencvDemo059
# from OpencvDemo060 import OpencvDemo060

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
    elif demoId == 31:
        # 图像梯度–Sobel算子
        ret = OpencvDemo031()
    elif demoId == 32:
        # 图像梯度–更多梯度算子
        ret = OpencvDemo032()
    elif demoId == 33:
        # 图像梯度–拉普拉斯算子
        ret = OpencvDemo033()
    elif demoId == 34:
        # 图像锐化
        ret = OpencvDemo034()
    elif demoId == 35:
        # USM锐化增强算法
        ret = OpencvDemo035()
    elif demoId == 36:
        # Canny边缘检测器
        ret = OpencvDemo036()
    elif demoId == 37:
        # 图像金字塔
        ret = OpencvDemo037()
    elif demoId == 38:
        # 拉普拉斯金字塔
        ret = OpencvDemo038()
    elif demoId == 39:
        # 图像模板匹配
        ret = OpencvDemo039()
    elif demoId == 40:
        # 二值图像介绍
        ret = OpencvDemo040()
    elif demoId == 41:
        # OpenCV中的基本阈值操作
        ret = OpencvDemo041()
    elif demoId == 42:
        # OTSU二值寻找算法
        ret = OpencvDemo042()
    elif demoId == 43:
        # TRIANGLE二值寻找算法
        ret = OpencvDemo043()
    elif demoId == 44:
        # 自适应阈值算法
        ret = OpencvDemo044()
    elif demoId == 45:
        # 图像二值化与去噪
        ret = OpencvDemo045()
    elif demoId == 46:
        # 二值图像连通组件寻找
        ret = OpencvDemo046()
    elif demoId == 47:
        # 二值图像连通组件状态统计
        ret = OpencvDemo047()
    elif demoId == 48:
        # 二值图像分析—轮廓发现
        ret = OpencvDemo048()
    elif demoId == 49:
        # 二值图像分析—轮廓外接矩形
        ret = OpencvDemo049()
    elif demoId == 50:
        # 二值图像分析—轮廓面积与弧长
        ret = OpencvDemo050()
    elif demoId == 51:
        # 二值图像分析—使用轮廓逼近
        ret = OpencvDemo051()
    elif demoId == 52:
        # 二值图像分析—用几何矩计算轮廓中心与横纵比过滤
        ret = OpencvDemo052()
    elif demoId == 53:
        # 二值图像分析—Hu矩实现轮廓匹配
        ret = OpencvDemo053()
#     elif demoId == 54:
#         # XXX
#         ret = OpencvDemo054()
#     elif demoId == 55:
#         # XXX
#         ret = OpencvDemo055()
#     elif demoId == 56:
#         # XXX
#         ret = OpencvDemo056()
#     elif demoId == 57:
#         # XXX
#         ret = OpencvDemo057()
#     elif demoId == 58:
#         # XXX
#         ret = OpencvDemo058()
#     elif demoId == 59:
#         # XXX
#         ret = OpencvDemo059()
#     elif demoId == 60:
#         # XXX
#         ret = OpencvDemo060()
    else:
        print("The argument is invalid!")
        return -1
    
    return ret

if __name__ == "__main__":
    main(argv)

# end of file
