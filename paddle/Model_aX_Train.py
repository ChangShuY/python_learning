'''
文件名：Model_aX_Train.py
一个 y=ax 的线性网络的训练过程
源自 paddle官网 fluid1.5 的一个代码实例
https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/programming_guide/programming_guide.html#id2

CSY 2019-12-31 练习过程中有改动
'''
#!/usr/bin/python
#_*_ coding: utf-8 _*_

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
y_true = numpy.array([[2.0],[4.0],[6.0],[8.0]]).astype('float32') # y的取值

#定义网络
#输入层
x = fluid.data(name="x",shape=[-1,1],dtype='float32') #原文为 fluid.layers.data
y = fluid.layers.data(name="y",shape=[1],dtype='float32') #原文为 fluid.layers.data
#全连接层
y_predict = fluid.layers.fc(input=x,size=1,act=None)

#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
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
print outs
