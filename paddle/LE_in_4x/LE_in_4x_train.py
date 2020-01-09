#!/usr/bin/env python
# coding: utf-8

#一个求解4元一次方程的线性回归模型
# Linear Equation In Four Unknowns
# csy 2019-10-30
# https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html

#加载库
import paddle.fluid as fluid
import numpy as np

#生成数据, 奇怪：当group = 19和20时，越练误差越大
group = input('请输入一个正整数给group赋值：')
group = int(group)
np.random.seed(0)
outputs = np.random.randint(5,size=(group,4)) #生成group组，每组4个小于10的随机数
#outputs = np.random.randint(5, size=(10, 4))

res = []
for i in range(group):
    #假设方程式为 y=4a+6b+7c+2d，生成答案
    y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
    res.append([y])  # 当变量为array[][]时，对应的y值保存在res

# 定义数据
train_data = np.array(outputs).astype('float32')   # 训练数据使用10组随机生成的数据，将整型随机数改为浮点型
y_true = np.array(res).astype('float32')           # 对应的标准答案

# 定义网络
x = fluid.data(name="x",shape=[None,4],dtype='float32')
y = fluid.data(name="y",shape=[None,1],dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)

# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

# 定义优化
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
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
        
# 存储训练结果
params_dirname = 'model'
fluid.io.save_inference_model(params_dirname,['x'],[y_predict],exe)

