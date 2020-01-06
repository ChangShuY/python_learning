#!/usr/bin/python
#_*_ coding: utf-8 _*_

'''
文件名：Model_aX_Train.py
一个 y=ax 的线性网络的训练过程
源自 paddle官网 fluid1.5 的一个代码实例
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/programming_guide/programming_guide.html#id2

CSY 2019-12-31 练习过程中有改动

给定一组数据 <X,Y>，求解出系数a，使得 y= a*x，
其中X,Y均为一维张量。最终网络可以依据输入x，准确预测出y_predict。
2020-1-2 用4个数据训练100次后的结果如下：
当x=1时，y_predict = 2.1513414
当x=2时，y_predict = 4.0733356
当x=3时，y_predict = 5.9953294
当x=4时，y_predict = 7.9173236
均方差为 0.00878489

'''
#加载库
import paddle.fluid as fluid
import numpy

#定义训练数据
'''
已知：
当x=1时，y=2;
当x=2时，y=4;
当x=3时，y=6;
当x=4时，y=8
'''
train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32') # x的取值

#定义仅有正向传播的预测网络
#输入层
#x = fluid.data(name="x",shape=[-1,1],dtype='float32') #原文为 fluid.layers.data
x = fluid.layers.data(name="x",shape=[1],dtype='float32') #原文为 fluid.layers.data
#全连接层
y_predict = fluid.layers.fc(input=x,size=1,act=None)

#增加反向传播网络
#定义标签
y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32') # y的取值
#输入层
#y = fluid.data(name="y",shape=[-1,1],dtype='float32') #原文为 fluid.layers.data
y = fluid.layers.data(name="y",shape=[1],dtype='float32') #原文为 fluid.layers.data
#定义损失函数,评估预测结果的好坏
cost = fluid.layers.square_error_cost(input=y_predict,label=y) # 采用方差函数
avg_cost = fluid.layers.mean(cost) # 采用方差函数
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01) # SGD优化器
sgd_optimizer.minimize(avg_cost) # 根据反向计算所得的梯度，更新权重，使得avg_cost最小

#参数初始化
cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

##开始训练，迭代100次
for i in range(100):
    outs = exe.run(
        feed={'x':train_data,'y':y_true},
        fetch_list=[y_predict.name,avg_cost.name])
#观察结果
print(outs)

# 存储训练后的模型
params_dirname = 'model'
fluid.io.save_inference_model(
    params_dirname, # dirname: 保存预测模型的文件夹
    ['x'], # feeded_var_names: 预测时所需提供数据的所有输入变量的名称
    [y_predict], # target_vars：模型的输出变量
    exe) # executor：模型的执行器

infer_exe = fluid.Executor(cpu) # 在cpu上运行
inference_scope = fluid.Scope() # 获取作用域变量

# 加载训练好的模型
'''
返回fetch_list=
inference_program -- load_inference_model 返回的预测Program
feed_target_names -- load_inference_model 返回的所有输入变量的名称
fetch_targets     -- load_inference_model 返回的输出变量
参数
params_dirname    -- 待加载模型的存储路径 
infer_exe         -- 运行模型的执行器
'''
params_dirname = 'model'
with fluid.scope_guard(inference_scope):
    [inference_program,feed_target_names,fetch_targets] =  fluid.io.load_inference_model(params_dirname, infer_exe)

# 生成测试数据, 模型的输入层设计为只接收float32
group = 4
#x_test = numpy.random.uniform(0,10.0,size=(group,1)) #生成8个不同的a, a>=0且a<10, a的类型float64 
#x_test = x_test.astype(numpy.float32) # 转换成float32
x_test = numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32') # x的取值

# 进行预测
results = infer_exe.run(inference_program,
                        feed={'x':x_test},
                        fetch_list=fetch_targets)
                        #fetch_list=[y_predict.name])
# 显示答案
for i in range(group):
    print("当x={x}时，预测结果为y={y}".format(x=x_test[i][0],y=results[i][0]))
