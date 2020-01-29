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
2. 改训练过程
3. 分为两种初始化程序
4. 模型保存到model1文件夹中

重要参考：DataFeeder https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/DataFeeder_cn.html#datafeeder

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

# 建立Reader
def reader():
    #def req_one_data(): # 注释了原文的此条语句
        for i in range(10):
            data_X = [i]        
            data_Y = [i * 10 + 3]
            data_X = np.array(data_X).reshape(1, 1).astype("float32")    
            data_Y = np.array(data_Y).reshape(1, 1).astype("float32")
            yield data_X, data_Y    # 使用yield来返回单条数据
    #return req_one_data    # 返回 req_one_data 这个变量名！可不是req_one_data() # 注释了原文的此条语句

# 初始化项目环境
# fluid.Program 默认有 default_startup_program 和 default_main_program
# 将 start_program 和 main_program 分开定义后，就可以用 program_guard 设置两个不同的程序空间
main_program = fluid.Program() # 空白程序框架
start_test = fluid.Program() # 空白的初始化程序,用于测试
start_train = fluid.default_startup_program()  # 默认的初始化程序，用于训练。

# 定义 main_program 程序空间的变量，使用startup_program 进行初始化，此处因使用的空白的初始化程序，说明在此程序空间不需要初始化变量
with fluid.program_guard(main_program=main_program, startup_program=start_test): # startup_program 默认为 default_startup_program
    # 定义张量格式
    x = fluid.data(name="x", shape=[-1,1], dtype="float32") # 第一个参数-1表示每批可以喂任意多的题目。第二个参数1表示每题只有一个已知条件。
    y = fluid.data(name="y", shape=[-1,1], dtype="float32") # 第一个参数-1表示每批可以喂任意多的题目。第二个参数1表示每题只有一个数字表示的答案。

    # 定义神经网络
    out = fluid.layers.fc(input=x, size=1)

    # 定义损失函数
    loss = fluid.layers.square_error_cost(input=out, label=y) # 使用均方差损失函数进行计算损失
    avg_loss = fluid.layers.mean(loss)  # 对损失求平均

    # 定义优化器
    opt = fluid.optimizer.SGD(learning_rate=0.005)  # 使用随机梯度下降策略进行优化，学习率为0.01
    opt.minimize(avg_loss)  #拿到损失值后，进行反向传播

# 初始化环境
place = fluid.CPUPlace()  # 在CPU中运算
exe = fluid.Executor(place)  # 初始化执行器
exe.run(start_train)  # 准备执行框架

# 定义数据传输格式
train_reader = fluid.io.batch(reader=reader, batch_size=10) # 原文为paddle.batch
'''
train_feeder = fluid.DataFeeder(feed_list=[x, y],
                                place=place, 
                                program=main_program) # 默认为 default_main_program
'''

# 开始训练
for i in range(500): # 把刚才的5道题的题库，练习100遍！
    for data in train_reader():  #从题库里每次抽出一道题
        info = exe.run(
            program = main_program,
            feed = train_feeder.feed(data),
            fetch_list=[loss]) 
    if i % 50 == 0:
        print(i,info[0])

# 保存训练后的模型：输入端的网络标号为x,输出端的网络标号为out
fluid.io.save_inference_model(dirname="model1",
                              feeded_var_names=["x"],
                              target_vars=[out], 
                              executor=exe,
                              main_program = main_program)