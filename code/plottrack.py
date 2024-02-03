# coding=UTF-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pylab
import math




def loadData(fileName):
    inFile = open(fileName, 'r')  
   
    #t = []
    diff_x = []
    diff_y = []
    #sum_xy = []

    for line in inFile:
        trainingSet = line.split(',')  
        #t.append(float(trainingSet[0]))  
        diff_x.append(float(trainingSet[0]))  
        diff_y.append(float(trainingSet[1]))  
    # sum_xy.append( math.sqrt(float(trainingSet[2])*float(trainingSet[2]) + float(trainingSet[1])*float(trainingSet[1])) )

    return (diff_x, diff_y)


(diff_x, diff_y) = loadData('/media/hp/new/yzx/1/siamban-master/siamban-master/round1')


pylab.figure(1)
          

pylab.plot(diff_x, diff_y, linewidth=1) 

pylab.legend(loc='best', fontsize=20)  

pylab.xlabel("x", fontsize=20)  
pylab.ylabel("y", fontsize=20)  
pylab.xticks(fontsize=20) 
pylab.yticks(fontsize=20)

pylab.show()  

