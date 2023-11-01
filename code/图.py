# coding=UTF-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pylab
import math


# 从.txt中读取数据 的函数

def loadData(fileName):
    inFile = open(fileName, 'r')  # 以只读方式打开某filename文件
    # 定义2个空的list，用来存放文件中的数据
    #t = []
    diff_x = []
    diff_y = []
    #sum_xy = []

    for line in inFile:
        trainingSet = line.split(',')  # 对于上面数据每一行，按' '把数据分开，这里是分成两部分
        #t.append(float(trainingSet[0]))  # 第一部分，即文件中的第一列数据逐一添加到list t中
        diff_x.append(float(trainingSet[0]))  # 第二部分，即文件中的第1列数据逐一添加到list accx中
        diff_y.append(float(trainingSet[1]))  # 第三部分，即文件中的第2列数据逐一添加到list accy中
    # sum_xy.append( math.sqrt(float(trainingSet[2])*float(trainingSet[2]) + float(trainingSet[1])*float(trainingSet[1])) )

    return (diff_x, diff_y)


(diff_x, diff_y) = loadData('/media/hp/new/yzx/1/siamban-master/siamban-master/round1')
#(t1, diff_x1, diff_y1) = loadData('/home/jht/VIns_Code/GVINS/output/HPL_NO_fault.txt')

pylab.figure(1)
# pylab.figure(figsize=(8,6))              # 定义图的大小

pylab.plot(diff_x, diff_y, linewidth=1)  # 线的粗细、颜色、标签
#pylab.plot(diff_y, linestyle='--', label='Y', color="blue", marker='o',linewidth=1)  # 线的粗细、颜色、标签
pylab.legend(loc='best', fontsize=20)  # pylab.legend 放在 plot后面才能显示label, 并且设置标签的字体大小

pylab.xlabel("x", fontsize=20)  # 横轴
pylab.ylabel("y", fontsize=20)  # 纵轴
pylab.xticks(fontsize=20)  # 设置【坐标】标签字体大小
pylab.yticks(fontsize=20)

# pylab.title("Example")                             # 标题
#pylab.xlim([0,5000])                               # X轴数据长度设定
#pylab.ylim([0,1])                                  # y轴数据长度设定

pylab.show()  # 让绘制的图像在屏幕上显示出来

