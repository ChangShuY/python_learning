'''
来源：
使用PaddlePaddle创建属于你的第一个深度学习程序 -- Paddle Paddle 飞桨 入门实战教程 (1)
https://www.jianshu.com/p/a8329ec94059
Reader、Program、Scope使用 -- Paddle Paddle 飞桨 入门实战教程 (2)
https://www.jianshu.com/p/1707d2a75f75

运行环境：Ubuntu19,python3.7.0,paddle1.6.0

训练一个全连接层，能够预测一元一次线性方程
程序源于10xAnd3_train0.py

csy 第1次学习改编于 2020-1-23
1. 改为reader读取数据

运行结果：
0 [[4250.303]]
50 [[0.07298885]]
100 [[0.00996768]]
150 [[0.00136186]]
200 [[0.00018609]]
250 [[2.543215e-05]]
300 [[3.4939148e-06]]
350 [[4.7148205e-07]]
400 [[6.338814e-08]]
450 [[9.837095e-09]]

'''

#!/usr/bin/env python
# coding: utf-8

import paddle.fluid as fluid
import numpy as np

def reader():
    def req_one_data():
        for i in range(10):
            data_X = [i]        
            data_Y = [i * 10 + 3]
            data_X = np.array(data_X).reshape(1, 1).astype("float32")    
            data_Y = np.array(data_Y).reshape(1, 1).astype("float32")
            yield data_X, data_Y    # 使用yield来返回单条数据
    return req_one_data    # 返回 req_one_data 这个变量名！可不是req_one_data()

x = fluid.data(name="x", shape=[-1,1], dtype="float32") # 第一个参数-1表示每批可以喂任意多的题目。第二个参数1表示每题只有一个已知条件。
y = fluid.data(name="y", shape=[-1,1], dtype="float32") # 第一个参数-1表示每批可以喂任意多的题目。第二个参数1表示每题只有一个数字表示的答案。

out = fluid.layers.fc(input=x, size=1)

loss = fluid.layers.square_error_cost(input=out, label=y) # 使用均方差损失函数进行计算损失
avg_loss = fluid.layers.mean(loss)  # 对损失求平均(至于为什么要定义这句，下一节会介绍这个问题)

opt = fluid.optimizer.SGD(learning_rate=0.005)  # 使用随机梯度下降策略进行优化，学习率为0.01
opt.minimize(avg_loss)  #拿到损失值后，进行反向传播

place = fluid.CPUPlace()  # 初始化CPU运算环境
start = fluid.default_startup_program()  # 初始化训练框架环境
exe = fluid.Executor(place)  # 初始化执行器
exe.run(start)  # 准备执行框架

# 定义数据传输格式
train_reader = paddle.batch(reader=reader(), batch_size=10)
train_feeder = fluid.DataFeeder(feed_list=[x, y], place=place, program=hi_ai_program)

for i in range(500): # 把刚才的5道题的题库，练习100遍！
    for x_, y_ in zip(data_X, data_Y):  #从题库里每次抽出一道题
        x_ = np.array(x_).reshape(1, 1).astype("float32")  # 将抽出的题目转换成numpy对象
        y_ = np.array(y_).reshape(1, 1).astype("float32") # 将抽出的答案转换成numpy对象
        info = exe.run(feed={"x": x_, "y": y_}, fetch_list=[loss]) 
    if i % 50 == 0:
        print(i,info[0])

# 保存训练后的模型：输入端的网络标号为x,输出端的网络标号为out
fluid.io.save_inference_model(dirname="model", feeded_var_names=["x"], target_vars=[out], executor=exe)