#!/usr/bin/env python
# coding: utf-8

#一个求解4元一次方程的线性回归模型
# Linear Equation In Four Unknowns
# csy 2020-1-7 使用默认作用域
# 抄录于百度飞浆网站 https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html#id5

#加载库
import paddle.fluid as fluid
import numpy as np

# 开始预测
cpu = fluid.CPUPlace()
infer_exe = fluid.Executor(cpu)
inference_scope = fluid.Scope()

# 加载训练好的模型
params_dirname = 'result'
with fluid.scope_guard(inference_scope):
    [inference_program,feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(params_dirname,infer_exe)
    
    # 生成测试数据
    test = np.array([[[9],[5],[2],[10]]]).astype('float32')

    # 进行预测
    results = infer_exe.run(inference_program,
                        feed={'x':test},
                        fetch_list=fetch_targets)

    # 给出题目为 【9,5,2,10】 输出y=4*9+6*5+7*2+10*2的值
    print ("9a+5b+2c+10d={}".format(results[0][0]))