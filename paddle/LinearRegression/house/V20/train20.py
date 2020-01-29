##一个经典的房价预测的线性回归（Linear Regression [1]）模型训练程序
##https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basics/fit_a_line/README.cn.html
##CSY 2019-11-14 v20
## v10运行结果的最后二行：
## Figure(640x480)
## train cost, Step 2090, Cost 21.099422

## v20运行结果的二行：
## train cost,Step 640,Cost 90.977318
## train cost,Step 650,Cost 55.864998

##CSY 2019-11-22 调试通过，见 Run_train20.py.ipynb

#引入必要的库
from __future__ import print_function
#import paddle
import paddle.fluid as fluid #fluid v1.6
import numpy
import math
import sys

#准备训练数据
train_data=numpy.load("train_data.npy")
#准备测试数据
test_data=numpy.load("test_data.npy")

def reader_creator(train_data):  
    def reader():  
        for d in train_data:
            # yield 用法见 https://blog.csdn.net/mieleizhi0522/article/details/82142856
            # d每次取train_data的一行
            # 转换成两个一维张量。data[:-1]得到该行的去掉后尾的数据；d[-1:]取最后一个数
            yield d[:-1], d[-1:]  
    return reader

BATCH_SIZE = 20 #与全连接层的输出有关。全连接层是按批进行处理的。

#训练数据的阅读器，一次缓冲全部数据，并将记录的顺序全部打乱，每次读取20条记录
train_reader = fluid.io.batch(
    fluid.io.shuffle(
        reader_creator(train_data), buf_size=450),
        batch_size=BATCH_SIZE)

#测试数据的阅读器，一次缓冲全部数据
test_reader = fluid.io.batch(
    fluid.io.shuffle(
        reader_creator(test_data), buf_size=110),
        batch_size=BATCH_SIZE)

#n=0
#for i in test_reader():
#    n += 1
#    print(n,i)
 
#配置训练程序
#定义一个从输入到输出的全连接层
x = fluid.data(name='x', shape=[-1,13], dtype='float32') # 定义输入样本的形状和数据类型
#x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.data(name='y', shape=[-1,1], dtype='float32') # 定义标签的形状和数据类型
#y = fluid.layers.data(name='y', shape=[1], dtype='float32') 
y_predict = fluid.layers.fc(input=x, size=1, act=None) # 连接输入和输出的全连接层，输出预测值

#print('x,y=fluid.data()')
#print('x,y=fluid.layers.data()')

main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 利用输出的预测数据和标签数据估计方差
avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失

#配置优化算法
#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001) #随机梯度下降算法的优化器
sgd_optimizer.minimize(avg_loss)

#定义运算场所
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() # 指明executor的执行场所 place = fluid.CPUPlace()

###executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，调用run(...)执行program。
exe = fluid.Executor(place) # 训练执行器

#创建训练过程
num_epochs = 100 # 训练全部数据*100遍

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失


#初始化执行器
#%matplotlib inline # 在jupyter notebook中用%run train20.py时，此句报错。但现运行此句，再%run train20.py，可画出曲线。
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y]) # 喂入样本x和标签y
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
#从paddle.utils.plot模块中引入Ploter函数,未搜索到该模块说明，也无该函数说明
from paddle.utils.plot import Ploter
#定义plot_prompt为一幅图，该图画出 train_prompt 和 test_prompt 曲线
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place) # 测试执行器exe_test和训练执行器exe使用相同的执行器

#训练主循环
for pass_id in range(num_epochs):
    #train_reader()每递进一次（读入一批），就是从缓冲器中随机读取20条记录，共有21批数据，第21批只有4条记录
    #每条记录由[13]和[1]组成。
    for data_train in train_reader():
        #本循环体循环次数=100次*21批
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0: # 每10个批次记录并输出一下训练损失
            # train_prompt曲线上增加一点(step,avg_loss_value[0])
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            # 刷新显示 plot_prompt 图
            plot_prompt.plot()
            # 刷新显示的图，会覆盖掉先前的print内容。
            print("%s, Step %d, Cost %f" %
                  (train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name], # 输出损失算法的名字，我测试得出: ‘mean_3.tmp_0’
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            print("%s, Step %d, Cost %f" %
                  (test_prompt, step, test_metics[0]))
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break
                
        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
             fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)