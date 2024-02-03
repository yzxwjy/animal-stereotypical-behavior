import numpy as np
import matplotlib.pyplot as plt
# 输入数据
def loadData(filename):
    inFile = open(filename, 'r')
    x_data = []
    y_data = []
    for line in inFile:
        data = line.split(',')
        x_data.append(float(data[0]))
        y_data.append(float(data[1]))
    return (x_data, y_data)
(x_data, y_data) = loadData('round')

# 计算互相关系数
global r
r = np.corrcoef(x_data, y_data)[0, 1]

print("互相关系数:", r)
"""
plt.scatter(x_data, y_data, label=f'互相关系数 = {r:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x与y的互相关系数图')
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
"""
