import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# 生成示例时间序列数据 x 和数值数据 y，你需要用你自己的数据替换这里的示例数据
#num_points = 100
#x = np.linspace(0, 10, num_points)  # 示例时间序列数据
#y = np.sin(x) + np.random.normal(0, 0.2, num_points)  # 示例数值数据
def loaddata(filename):
    infile = open(filename, "r")
    diff_x = []
    diff_y = []

    for line in infile:
        trainset = line.split(' ')
        diff_x.append(float(trainset[0]))
        diff_y.append(float(trainset[1]))

    return (diff_x, diff_y)
(diff_x, diff_y) = loaddata('line')
# 计算 y 的自相关系数
max_lag = 385  # 最大延迟
#max_lag = 511
#max_lag=671
#autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full') / (np.std(y) ** 2 * len(y))
#autocorr = autocorr[num_points - 1:num_points + max_lag]
acf = np.correlate(diff_y - np.mean(diff_y), diff_y - np.mean(diff_y), mode='full') / (np.std(diff_y) ** 2 * len(diff_y))
acf = acf[386 - 1:386 + max_lag]
#acf = acf[512 - 1:512 + max_lag]
#acf = acf[672 - 1:672 + max_lag]
# 绘制自相关系数图
peaks,_ = scipy.signal.find_peaks(acf,height=0)
plt.figure(figsize=(12, 7))
plt.stem(np.arange(0, max_lag + 1), acf)
#plt.stem(acf)
#plt.stem(acf，linefmt='k',markerfmt='c')
plt.title('Circular curve of Autocorrelation',fontsize=20)
#plt.title('Straight line of Autocorrelation',fontsize=20)
#plt.title('8-shaped curve of Autocorrelation',fontsize=20)
plt.xlabel('Lag', fontsize=20)
plt.ylabel('ACF', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.grid(True)
plt.plot(peaks, acf[peaks], "o",color="k")
#plt.text(peaks,acf[peaks],"peaks")
plt.savefig('D:/yzx/siamban-master/1.png',bbox_inches='tight')
#plt.show()

#print(peaks)




