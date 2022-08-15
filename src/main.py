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
from OpencvDemo054 import OpencvDemo054
from OpencvDemo055 import OpencvDemo055
from OpencvDemo056 import OpencvDemo056
from OpencvDemo057 import OpencvDemo057
from OpencvDemo058 import OpencvDemo058
from OpencvDemo059 import OpencvDemo059
from OpencvDemo060 import OpencvDemo060
from OpencvDemo061 import OpencvDemo061
from OpencvDemo062 import OpencvDemo062
from OpencvDemo063 import OpencvDemo063
from OpencvDemo064 import OpencvDemo064
from OpencvDemo065 import OpencvDemo065
from OpencvDemo066 import OpencvDemo066
from OpencvDemo067 import OpencvDemo067
from OpencvDemo068 import OpencvDemo068
from OpencvDemo069 import OpencvDemo069
from OpencvDemo070 import OpencvDemo070
from OpencvDemo071 import OpencvDemo071
from OpencvDemo072 import OpencvDemo072
from OpencvDemo073 import OpencvDemo073
from OpencvDemo074 import OpencvDemo074
from OpencvDemo075 import OpencvDemo075
from OpencvDemo076 import OpencvDemo076
from OpencvDemo077 import OpencvDemo077
from OpencvDemo078 import OpencvDemo078
from OpencvDemo079 import OpencvDemo079
from OpencvDemo080 import OpencvDemo080
from OpencvDemo081 import OpencvDemo081
from OpencvDemo082 import OpencvDemo082
from OpencvDemo083 import OpencvDemo083
from OpencvDemo084 import OpencvDemo084
from OpencvDemo085 import OpencvDemo085
from OpencvDemo086 import OpencvDemo086
from OpencvDemo087 import OpencvDemo087
from OpencvDemo088 import OpencvDemo088
from OpencvDemo089 import OpencvDemo089
from OpencvDemo090 import OpencvDemo090
from OpencvDemo091 import OpencvDemo091
from OpencvDemo092 import OpencvDemo092
from OpencvDemo093 import OpencvDemo093
from OpencvDemo094 import OpencvDemo094
from OpencvDemo095 import OpencvDemo095
from OpencvDemo096 import OpencvDemo096
from OpencvDemo097 import OpencvDemo097
from OpencvDemo098 import OpencvDemo098
from OpencvDemo099 import OpencvDemo099
from OpencvDemo100 import OpencvDemo100
from OpencvDemo101 import OpencvDemo101
from OpencvDemo102 import OpencvDemo102
from OpencvDemo103 import OpencvDemo103
from OpencvDemo104 import OpencvDemo104
from OpencvDemo105 import OpencvDemo105
from OpencvDemo106 import OpencvDemo106
from OpencvDemo107 import OpencvDemo107
from OpencvDemo108 import OpencvDemo108
from OpencvDemo109 import OpencvDemo109
from OpencvDemo110 import OpencvDemo110
from OpencvDemo111 import OpencvDemo111
from OpencvDemo112 import OpencvDemo112
from OpencvDemo113 import OpencvDemo113
from OpencvDemo114 import OpencvDemo114
from OpencvDemo115 import OpencvDemo115
from OpencvDemo116 import OpencvDemo116
from OpencvDemo117 import OpencvDemo117
from OpencvDemo118 import OpencvDemo118
from OpencvDemo119 import OpencvDemo119
from OpencvDemo120 import OpencvDemo120
from OpencvDemo121 import OpencvDemo121
from OpencvDemo122 import OpencvDemo122
from OpencvDemo123 import OpencvDemo123
from OpencvDemo124 import OpencvDemo124
from OpencvDemo125 import OpencvDemo125
from OpencvDemo126 import OpencvDemo126
from OpencvDemo127 import OpencvDemo127
from OpencvDemo128 import OpencvDemo128
from OpencvDemo129 import OpencvDemo129
from OpencvDemo130 import OpencvDemo130
from OpencvDemo131 import OpencvDemo131
from OpencvDemo132 import OpencvDemo132
from OpencvDemo133 import OpencvDemo133
from OpencvDemo134 import OpencvDemo134
from OpencvDemo135 import OpencvDemo135
from OpencvDemo136 import OpencvDemo136
from OpencvDemo137 import OpencvDemo137
from OpencvDemo138 import OpencvDemo138
# from OpencvDemo139 import OpencvDemo139
# from OpencvDemo140 import OpencvDemo140

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
    elif demoId == 54:
        # 二值图像分析—对轮廓圆与椭圆拟合
        ret = OpencvDemo054()
    elif demoId == 55:
        # 二值图像分析—凸包检测
        ret = OpencvDemo055()
    elif demoId == 56:
        # 二值图像分析—直线拟合与极值点寻找
        ret = OpencvDemo056()
    elif demoId == 57:
        # 二值图像分析—点多边形测试
        ret = OpencvDemo057()
    elif demoId == 58:
        # 二值图像分析—寻找最大内接圆
        ret = OpencvDemo058()
    elif demoId == 59:
        # 二值图像分析—霍夫直线检测
        ret = OpencvDemo059()
    elif demoId == 60:
        # 二值图像分析—霍夫直线检测二
        ret = OpencvDemo060()
    elif demoId == 61:
        # 二值图像分析—霍夫圆检测
        ret = OpencvDemo061()
    elif demoId == 62:
        # 图像形态学—膨胀与腐蚀
        ret = OpencvDemo062()
    elif demoId == 63:
        # 图像形态学—膨胀与腐蚀二
        ret = OpencvDemo063()
    elif demoId == 64:
        # 图像形态学–开操作
        ret = OpencvDemo064()
    elif demoId == 65:
        # 图像形态学—闭操作
        ret = OpencvDemo065()
    elif demoId == 66:
        # 图像形态学—开闭操作时候结构元素应用演示
        ret = OpencvDemo066()
    elif demoId == 67:
        # 图像形态学—顶帽操作
        ret = OpencvDemo067()
    elif demoId == 68:
        # 图像形态学—黑帽操作
        ret = OpencvDemo068()
    elif demoId == 69:
        # 图像形态学—图像梯度
        ret = OpencvDemo069()
    elif demoId == 70:
        # 形态学应用—用基本梯度实现轮廓分析
        ret = OpencvDemo070()
    elif demoId == 71:
        # 形态学操作—击中击不中
        ret = OpencvDemo071()
    elif demoId == 72:
        # 二值图像分析—缺陷检测一
        ret = OpencvDemo072()
    elif demoId == 73:
        # 二值图像分析—缺陷检测二
        ret = OpencvDemo073()
    elif demoId == 74:
        # 二值图像分析—提取最大轮廓与编码关键点
        ret = OpencvDemo074()
    elif demoId == 75:
        # 图像去水印/修复
        ret = OpencvDemo075()
    elif demoId == 76:
        # 图像透视变换应用
        ret = OpencvDemo076()
    elif demoId == 77:
        # 视频读写与处理
        ret = OpencvDemo077()
    elif demoId == 78:
        # 识别与跟踪视频中的特定颜色对象
        ret = OpencvDemo078()
    elif demoId == 79:
        # 视频分析—背景/前景提取
        ret = OpencvDemo079()
    elif demoId == 80:
        # 视频分析—背景消除与前景ROI提取
        ret = OpencvDemo080()
    elif demoId == 81:
        # 角点检测—Harris角点检测
        ret = OpencvDemo081()
    elif demoId == 82:
        # 角点检测—Shi-Tomasi角点检测
        ret = OpencvDemo082()
    elif demoId == 83:
        # 角点检测–亚像素级别角点检测
        ret = OpencvDemo083()
    elif demoId == 84:
        # 视频分析—移动对象的KLT光流跟踪算法之一
        ret = OpencvDemo084()
    elif demoId == 85:
        # 视频分析-移动对象的KLT光流跟踪算法之二
        ret = OpencvDemo085()
    elif demoId == 86:
        # 视频分析–稠密光流分析
        ret = OpencvDemo086()
    elif demoId == 87:
        # 视频分析—基于帧差法实现移动对象分析
        ret = OpencvDemo087()
    elif demoId == 88:
        # 视频分析—基于均值迁移的对象移动分析
        ret = OpencvDemo088()
    elif demoId == 89:
        # 视频分析—基于连续自适应均值迁移的对象移动分析
        ret = OpencvDemo089()
    elif demoId == 90:
        # 视频分析—对象移动轨迹绘制
        ret = OpencvDemo090()
    elif demoId == 91:
        # 对象检测—HAAR级联检测器使用
        ret = OpencvDemo091()
    elif demoId == 92:
        # 对象检测—HAAR特征介绍
        ret = OpencvDemo092()
    elif demoId == 93:
        # 对象检测—LBP特征介绍
        ret = OpencvDemo093()
    elif demoId == 94:
        # ORB FAST特征关键点检测
        ret = OpencvDemo094()
    elif demoId == 95:
        # BRIEF特征描述子匹配
        ret = OpencvDemo095()
    elif demoId == 96:
        # 描述子匹配
        ret = OpencvDemo096()
    elif demoId == 97:
        # 基于描述子匹配的已知对象定位
        ret = OpencvDemo097()
    elif demoId == 98:
        # SIFT特征提取—关键点提取
        ret = OpencvDemo098()
    elif demoId == 99:
        # SIFT特征提取—描述子生成
        ret = OpencvDemo099()
    elif demoId == 100:
        # HOG特征与行人检测
        ret = OpencvDemo100()
    elif demoId == 101:
        # HOG特征描述子—多尺度检测
        ret = OpencvDemo101()
    elif demoId == 102:
        # HOG特征描述子—提取描述子
        ret = OpencvDemo102()
    elif demoId == 103:
        # HOG特征描述子—使用描述子特征生成样本数据
        ret = OpencvDemo103()
    elif demoId == 104:
        # SVM线性分类器
        ret = OpencvDemo104()
    elif demoId == 105:
        # HOG特征描述子—使用HOG进行对象检测
        ret = OpencvDemo105()
    elif demoId == 106:
        # AKAZE特征与描述子
        ret = OpencvDemo106()
    elif demoId == 107:
        # BRISK特征提取与描述子匹配
        ret = OpencvDemo107()
    elif demoId == 108:
        # 特征提取之关键点检测—GFTTDetector
        ret = OpencvDemo108()
    elif demoId == 109:
        # BLOB特征分析—SimpleBlobDetector使用
        ret = OpencvDemo109()
    elif demoId == 110:
        # KMeans数据分类
        ret = OpencvDemo110()
    elif demoId == 111:
        # KMeans图像分割
        ret = OpencvDemo111()
    elif demoId == 112:
        # KMeans图像分割—背景替换
        ret = OpencvDemo112()
    elif demoId == 113:
        # KMeans图像分割—主色彩提取
        ret = OpencvDemo113()
    elif demoId == 114:
        # KNN算法介绍
        ret = OpencvDemo114()
    elif demoId == 115:
        # KNN算法应用
        ret = OpencvDemo115()
    elif demoId == 116:
        # 决策树算法 介绍与使用
        ret = OpencvDemo116()
    elif demoId == 117:
        # 图像均值漂移分割
        ret = OpencvDemo117()
    elif demoId == 118:
        # GrabCut图像分割
        ret = OpencvDemo118()
    elif demoId == 119:
        # GrabCut图像分割—背景替换
        ret = OpencvDemo119()
    elif demoId == 120:
        # 二维码检测与识别
        ret = OpencvDemo120()
    elif demoId == 121:
        # OpenCV DNN 获取导入模型各层信息
        ret = OpencvDemo121()
    elif demoId == 122:
        # OpenCV DNN 实现图像分类
        ret = OpencvDemo122()
    elif demoId == 123:
        # OpenCV DNN 为模型运行设置目标设备与计算后台
        ret = OpencvDemo123()
    elif demoId == 124:
        # OpenCV DNN 基于SSD实现对象检测
        ret = OpencvDemo124()
    elif demoId == 125:
        # OpenCV DNN 基于SSD实现实时视频检测
        ret = OpencvDemo125()
    elif demoId == 126:
        # OpenCV DNN 基于残差网络的人脸检测
        ret = OpencvDemo126()
    elif demoId == 127:
        # OpenCV DNN 基于残差网络的视频人脸检测
        ret = OpencvDemo127()
    elif demoId == 128:
        # OpenCV DNN 直接调用TensorFlow的导出模型
        ret = OpencvDemo128()
    elif demoId == 129:
        # OpenCV DNN 调用OpenPose模型实现姿态评估
        ret = OpencvDemo129()
    elif demoId == 130:
        # OpenCV DNN 支持YOLO对象检测网络运行
        ret = OpencvDemo130()
    elif demoId == 131:
        # OpenCV DNN 支持YOLOv3-tiny版本实时对象检测
        ret = OpencvDemo131()
    elif demoId == 132:
        # OpenCV DNN 单张与多张图像的推断
        ret = OpencvDemo132()
    elif demoId == 133:
        # OpenCV DNN 图像彩色化模型使用
        ret = OpencvDemo133()
    elif demoId == 134:
        # OpenCV DNN ENet实现图像分割
        ret = OpencvDemo134()
    elif demoId == 135:
        # OpenCV DNN 实时快速的图像风格迁移
        ret = OpencvDemo135()
    elif demoId == 136:
        # OpenCV DNN 解析网络输出结果
        ret = OpencvDemo136()
    elif demoId == 137:
        # OpenCV DNN 实现性别与年龄预测
        ret = OpencvDemo137()
    elif demoId == 138:
        # OpenCV DNN 使用OpenVINO加速
        ret = OpencvDemo138()
    # elif demoId == 139:
    #     # 案例：识别0～9印刷体数字—Part1
    #     ret = OpencvDemo139()
    # elif demoId == 140:
    #     # 案例：识别0～9印刷体数字—Part2
    #     ret = OpencvDemo140()
    else:
        print("The argument is invalid!")
        return -1
    
    return ret

if __name__ == "__main__":
    main(argv)

# end of file
