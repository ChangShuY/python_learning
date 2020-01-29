##房价预测数据的前期处理
##CSY 2019-11-13
##要求输入数据在工作目录下的housing.data文件中
##输出归一化数据： 工作目录下的 data1.txt,可用numpy.fromfile('data1.txt', sep=' ')加载
##输出训练集数据：工作目录下的 train_data.npy,可用numpy.load("train_data.npy")加载
##输出训练集数据：工作目录下的 test_data.npy,可用numpy.load("test_data.npy")加载

#引入必要的库
import numpy

#定义数据提供器
#feature_names = [
#    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', #'DIS', 'RAD', 'TAX',
#    'PTRATIO', 'B', 'LSTAT', 'convert'
#]
#feature_num = len(feature_names) #14列
feature_num = 14
data = numpy.fromfile('housing.data', sep=' ') # 从文件中读取原始数据

#对数据进行预处理
data = data.reshape(data.shape[0] // feature_num, feature_num) # 一大串数据转换为14列一行，抛弃不足一行的数据
#归一化
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
for i in range(feature_num-1): # i的取值从0到12，不包含13(最后一列为单价列，不做处理)
    #减去平均值，除以取值范围
   data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) 
data.tofile('data1.txt',sep=' ') #保存归一化数据

ratio = 0.8 # 训练集和验证集的划分比例
offset = int(data.shape[0]*ratio)
train_data = data[:offset] #前80%为训练集
test_data = data[offset:]  #后20%为验证集
numpy.save('train_data.npy',train_data) #保存训练集
numpy.save('test_data.npy',test_data) #保存测试集

#train_data.tofile('train_data.txt',sep=' ') #保存训练集
#test_data.tofile('test_data.txt',sep=' ')   #保存测试集
