'''
运行环境：Ubuntu19,python3.7.0,paddle1.6.0
csy 2020-1-10 源于
https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html

1.fluid.layers.data 改为 fluid.data
2.改为用键盘输入数字确定组数，生成训练数据
3.改为用键盘输入数字确定生成的训练数据范围
4.改为用键盘输入数字确定随机种子
5.随机产生浮点数的训练数据
6.打印出训练数据
7.不保存模型
8.删除测试部分
9.学习率改为0.01

运行结果:
group  data_range   random_seed 误差随训练次数的变化data_range
18       5            0                   出现nan
18       5            1                   出现nan
18       5            2                   出现nan
10       5            0                   出现nan
'''

#!/usr/bin/env python
# coding: utf-8

#加载库
import paddle.fluid as fluid
import numpy as np

#生成数据
group = input('请输入一个正整数确定生成数据的数量：')
group = int(group)
data_range = input('请输入一个正整数确定生成训练数据的取值范围：')
float_range = float(data_range)
random_seed = input('请输入一个正整数作为随机数的种子：')

np.random.seed(int(random_seed))
outputs =np.random.uniform(1,float_range,size=(group,4)) #float64

res = []  #生成一个空list
for i in range(group):
    #假设方程式为 y=4a+6b+7c+2d，生成答案
    y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
    print("{}: {}=({},{},{},{})".format(i,y,outputs[i][0],outputs[i][1],outputs[i][2],outputs[i][3]))
    res.append([y])  # 当变量为array[][]时，对应的y值保存在res

# 定义数据
train_data = np.array(outputs).astype('float32')   # 训练数据使用10组随机生成的数据，将整型随机数改为浮点型
y_true = np.array(res).astype('float32')           # 对应的标准答案
print(train_data.shape,y_true.shape)

# 定义网络
x = fluid.data(name="x",shape=[None,4],dtype='float32')
y = fluid.data(name="y",shape=[None,1],dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)

# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

# 定义优化
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

## 开始训练，迭代500次
for i in range(500):
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost.name])
    if i%50==0:
        print('iter={:.0f},cost={}'.format(i,outs[1][0]))
