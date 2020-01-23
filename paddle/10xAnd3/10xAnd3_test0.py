'''
来源：
使用PaddlePaddle创建属于你的第一个深度学习程序 -- Paddle Paddle 飞桨 入门实战教程 (1)
https://www.jianshu.com/p/a8329ec94059

运行环境：Ubuntu19,python3.7.0,paddle1.6.0

使用10xAnd3_train0.py保存在model文件夹中的模型进行预测

csy 第1次学习改编于 2020-1-23
标准答案为：1003，5673，203
运行结果：
[[1003.0006]]
[[5673.004]]
[[203.00006]]

'''

#!/usr/bin/env python
# coding: utf-8

import paddle.fluid as fluid
import numpy as np

# 定义测试数据
test_data = [[100], [567], [20]]

# 初始化预测环境
exe = fluid.Executor(place=fluid.CPUPlace())

# 读取模型
Program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname="model",
                                                                          executor=exe)
# 开始预测
for x_ in test_data:
    x_ = np.array(x_).reshape(1, 1).astype("float32")
    out = exe.run(program=Program, feed={feed_target_names[0]: x_}, fetch_list=fetch_targets)
    print(out[0])