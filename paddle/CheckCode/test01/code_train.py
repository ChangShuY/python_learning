'''
来源：
PaddlePaddle 极简入门实践四：简单验证码识别
https://www.jianshu.com/p/df98fcc832ed?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=qq
学习改编自：train_easy.py

运行环境：Ubuntu19,python3.7.0,paddle1.6.0
csy 2020-1-12 

数据集介绍：
data文件夹中从1.jpg--2000.jpg共2000张个位数的验证码图片。
图片尺寸为：宽=15个像素，高=30个像素
其中1.jpg--1500.jpg作为训练集

data文件夹中的ocrData.txt是标签文件，其格式为：
第1个字符        1.jpg的标签‘4’
第2个字符        2.jpg的标签'8'
...
第n个字符        n.jpg的标签（0<n<20001,整数）
...
第2000个字符  2000.jpg的标签‘4’

原程序运行时间约5分钟

CSY 2020-1-15 运行时间3分30秒
运行结果：
训练后文件夹路径../test01/test.inference.model
当前训练进度百分比为：0.0%准确率为[0.04266667]
当前训练进度百分比为：2.0%准确率为[0.27933332]
当前训练进度百分比为：4.0%准确率为[0.754]
当前训练进度百分比为：6.0%准确率为[0.91466665]
当前训练进度百分比为：8.0%准确率为[0.96533334]
当前训练进度百分比为：10%准确率为[0.9866667]
当前训练进度百分比为：12%准确率为[0.99266666]
当前训练进度百分比为：14%准确率为[0.9953333]
当前训练进度百分比为：16%准确率为[0.996]
当前训练进度百分比为：18%准确率为[0.996]
当前训练进度百分比为：20%准确率为[0.99666667]
当前训练进度百分比为：22%准确率为[0.99666667]
当前训练进度百分比为：24%准确率为[0.99666667]
当前训练进度百分比为：26%准确率为[0.99666667]
当前训练进度百分比为：28%准确率为[0.99666667]
当前训练进度百分比为：30%准确率为[0.99733335]
当前训练进度百分比为：32%准确率为[0.99733335]
当前训练进度百分比为：34%准确率为[0.99733335]
当前训练进度百分比为：36%准确率为[0.99733335]
当前训练进度百分比为：38%准确率为[0.99733335]
当前训练进度百分比为：40%准确率为[0.99733335]
当前训练进度百分比为：42%准确率为[0.99733335]
当前训练进度百分比为：44%准确率为[0.99733335]
当前训练进度百分比为：46%准确率为[0.99733335]
当前训练进度百分比为：48%准确率为[0.99733335]
当前训练进度百分比为：50%准确率为[0.99733335]
当前训练进度百分比为：52%准确率为[0.99733335]
当前训练进度百分比为：54%准确率为[0.99733335]
当前训练进度百分比为：56%准确率为[0.99733335]
当前训练进度百分比为：57%准确率为[0.99733335]
当前训练进度百分比为：60%准确率为[0.99733335]
当前训练进度百分比为：62%准确率为[0.99733335]
当前训练进度百分比为：64%准确率为[0.99733335]
当前训练进度百分比为：66%准确率为[0.99733335]
当前训练进度百分比为：68%准确率为[0.99733335]
当前训练进度百分比为：70%准确率为[0.99733335]
当前训练进度百分比为：72%准确率为[0.99733335]
当前训练进度百分比为：74%准确率为[0.99733335]
当前训练进度百分比为：76%准确率为[0.99733335]
当前训练进度百分比为：78%准确率为[0.99733335]
当前训练进度百分比为：80%准确率为[0.99733335]
当前训练进度百分比为：82%准确率为[0.99733335]
当前训练进度百分比为：84%准确率为[0.99733335]
当前训练进度百分比为：86%准确率为[0.99733335]
当前训练进度百分比为：88%准确率为[0.99733335]
当前训练进度百分比为：90%准确率为[0.99733335]
当前训练进度百分比为：92%准确率为[0.99733335]
当前训练进度百分比为：94%准确率为[0.99733335]
当前训练进度百分比为：96%准确率为[0.99733335]
当前训练进度百分比为：98%准确率为[0.99733335]

将训练次数改为20次,运行时间1分35秒，运行结果：
训练后文件夹路径../test01/test.inference.model
第0次训练：准确率为[0.15533334]； 误差为mean_0.tmp_0
第1次训练：准确率为[0.49]； 误差为mean_0.tmp_0
第2次训练：准确率为[0.832]； 误差为mean_0.tmp_0
第3次训练：准确率为[0.96]； 误差为mean_0.tmp_0
第4次训练：准确率为[0.98066664]； 误差为mean_0.tmp_0
第5次训练：准确率为[0.98733336]； 误差为mean_0.tmp_0
第6次训练：准确率为[0.9906667]； 误差为mean_0.tmp_0
第7次训练：准确率为[0.994]； 误差为mean_0.tmp_0
第8次训练：准确率为[0.994]； 误差为mean_0.tmp_0
第9次训练：准确率为[0.9946667]； 误差为mean_0.tmp_0
第10次训练：准确率为[0.9953333]； 误差为mean_0.tmp_0
第11次训练：准确率为[0.99666667]； 误差为mean_0.tmp_0
第12次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第13次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第14次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第15次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第16次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第17次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第18次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0
第19次训练：准确率为[0.99733335]； 误差为mean_0.tmp_0

将训练次数改为15次,运行时间1分5秒，运行结果：
训练后文件夹路径../test01/test.inference.model
第1次训练：准确率为[0.07933334]； 误差为[2.6051848]
第2次训练：准确率为[0.514]； 误差为[1.6225581]
第3次训练：准确率为[0.88]； 误差为[1.0804213]
第4次训练：准确率为[0.9626667]； 误差为[0.76219344]
第5次训练：准确率为[0.98466665]； 误差为[0.57082677]
第6次训练：准确率为[0.9906667]； 误差为[0.44942948]
第7次训练：准确率为[0.99266666]； 误差为[0.3681442]
第8次训练：准确率为[0.9946667]； 误差为[0.31086022]
第9次训练：准确率为[0.9953333]； 误差为[0.26872092]
第10次训练：准确率为[0.996]； 误差为[0.23662451]
第11次训练：准确率为[0.99666667]； 误差为[0.21145158]
第12次训练：准确率为[0.99666667]； 误差为[0.19123767]
第13次训练：准确率为[0.99666667]； 误差为[0.17468107]
第14次训练：准确率为[0.99666667]； 误差为[0.16088054]
第15次训练：准确率为[0.99666667]； 误差为[0.14921148]

将训练次数改为50次,运行时间3分，运行结果：
训练后文件夹路径../test01/test.inference.model
第2次训练：准确率为[0.30866668]； 误差为[2.0476341]
第3次训练：准确率为[0.67333335]； 误差为[1.4014692]
第4次训练：准确率为[0.91333336]； 误差为[0.97565544]
第5次训练：准确率为[0.98]； 误差为[0.7097404]
第7次训练：准确率为[0.9913333]； 误差为[0.43546355]
第8次训练：准确率为[0.99266666]； 误差为[0.36134282]
第9次训练：准确率为[0.994]； 误差为[0.3081387]
第10次训练：准确率为[0.9946667]； 误差为[0.26842377]
第12次训练：准确率为[0.996]； 误差为[0.21353094]
第13次训练：准确率为[0.99666667]； 误差为[0.19388297]
第14次训练：准确率为[0.99733335]； 误差为[0.17766777]
第15次训练：准确率为[0.99733335]； 误差为[0.16407396]
第17次训练：准确率为[0.99733335]； 误差为[0.14259072]
第18次训练：准确率为[0.99733335]； 误差为[0.13396461]
第19次训练：准确率为[0.99733335]； 误差为[0.12640622]
第20次训练：准确率为[0.99733335]； 误差为[0.11972978]
第22次训练：准确率为[0.99733335]； 误差为[0.10847701]
第23次训练：准确率为[0.99733335]； 误差为[0.10369383]
第24次训练：准确率为[0.99733335]； 误差为[0.09936562]
第25次训练：准确率为[0.99733335]； 误差为[0.09543066]
第27次训练：准确率为[0.99733335]； 误差为[0.08854138]
第28次训练：准确率为[0.99733335]； 误差为[0.08550993]
第29次训练：准确率为[0.99733335]； 误差为[0.08271274]
第30次训练：准确率为[0.99733335]； 误差为[0.08012443]
第32次训练：准确率为[0.99733335]； 误差为[0.07548705]
第33次训练：准确率为[0.99733335]； 误差为[0.07340176]
第34次训练：准确率为[0.99733335]； 误差为[0.07145207]
第35次训练：准确率为[0.99733335]； 误差为[0.0696243]
第37次训练：准确率为[0.99733335]； 误差为[0.06629235]
第38次训练：准确率为[0.99733335]； 误差为[0.06477022]
第39次训练：准确率为[0.99733335]； 误差为[0.06333372]
第40次训练：准确率为[0.99733335]； 误差为[0.06197532]
第42次训练：准确率为[0.99733335]； 误差为[0.05946824]
第43次训练：准确率为[0.99733335]； 误差为[0.05830897]
第44次训练：准确率为[0.99733335]； 误差为[0.05720645]
第45次训练：准确率为[0.99733335]； 误差为[0.05615664]
第47次训练：准确率为[0.99733335]； 误差为[0.05419997]
第48次训练：准确率为[0.99733335]； 误差为[0.05328702]
第49次训练：准确率为[0.99733335]； 误差为[0.05241415]
第50次训练：准确率为[0.99733335]； 误差为[0.05157859]

将训练次数改为50次,运行时间3分：
训练后文件夹路径../test01/test.inference.model
第1次训练：准确率为[0.132]； 误差为[2.6865091]
第6次训练：准确率为[0.9906667]； 误差为[0.4545013]
第11次训练：准确率为[0.99733335]； 误差为[0.21362607]
第16次训练：准确率为[0.99733335]； 误差为[0.14110917]
第21次训练：准确率为[0.99733335]； 误差为[0.10686199]
第26次训练：准确率为[0.99733335]； 误差为[0.08701857]
第31次训练：准确率为[0.99733335]； 误差为[0.07409438]
第36次训练：准确率为[0.99733335]； 误差为[0.06501875]
第41次训练：准确率为[0.99733335]； 误差为[0.05829536]
第46次训练：准确率为[0.99733335]； 误差为[0.05311361]

'''
import paddle.fluid as fluid # paddlepaddle 推出的最新的API
#import paddle # 有一些功能是fluid不具备的，还是需要paddle模块
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

# 提示 Font family ['sans-serif'] not found.
# 注释掉下面两句
# 绘图部分全部使用英文
#from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置一种用来显示中文的字体


path = "../"
params_dirname = path + "test01/test.inference.model"
print("训练后文件夹路径" + params_dirname)
'''
若不放心当前位置，可以查看当前路径
CSY 2020-1-14
import os
print(os.getcwd())
'''

# 参数初始化
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载数据
datatype = 'float32'
# r--只读方式打开;t--文本模式
with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read() #打开标签数据集文件. a是一个list, a[i]为第i个验证码的标签

'''
Image有以下几种模式，用字符串表示：
1 (1-bit pixels, black and white, stored with one pixel per byte)
L (8-bit pixels, black and white)
P (8-bit pixels, mapped to any other mode using a color palette)
RGB (3x8-bit pixels, true color)
RGBA (4x8-bit pixels, true color with transparency mask)
CMYK (4x8-bit pixels, color separation)
YCbCr (3x8-bit pixels, color video format)
Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
LAB (3x8-bit pixels, the L*a*b color space)
HSV (3x8-bit pixels, Hue, Saturation, Value color space)
I (32-bit signed integer pixels)
F (32-bit floating point pixels)
'''
def data_reader():
    '''
    使用PaddlePaddle中reader生成数据集列表
    def read() 和 return reader 配对使用，作为样本级的reader接口。
    https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/howto/prepare_data/reader_cn.html#id1
    函数每次返回一个由 yield 决定的样本数据项。
    '''
    def reader():
        for i in range(1, 1501):
            im = Image.open(path + "data/" + str(i) + ".jpg").convert('L') #8bit灰度模式
            #im.show() # 查看加载的图片
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            #print(im)
            '''
            图像归一化处理
            1. 将[0,255]映射到[0,1];
            2. 放大特征值；
            3. 平移到以原点为中心[-1,1]
            '''
            im = im / 255.0 * 2.0 - 1.0

            labelInfo = a[i - 1]
            yield im, labelInfo

    return reader


# 定义网络
#x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype=datatype)
x = fluid.data(name="x", shape=[-1, 1, 30, 15], dtype=datatype)
#label = fluid.layers.data(name='label', shape=[1], dtype='int64')
label = fluid.data(name='label', shape=[-1,1], dtype='int64')
'''
CNN,卷积神经网络
https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basics/image_classification/index.html#cnn
卷积层：执行卷积操作提取底层到高层的特征，发掘出图片局部关联性质和空间不变性质。
池化层：执行降采样操作，通过取卷积输出特征图中局部区块的最大值或者均值，可以过滤掉一些不重要的高频信息。
'''
def cnn(ipt):
    #print(ipt.shape)
    #二维卷积层 输入和输出格式=NCHW 即批尺寸、通道数、特征高度、特征宽度
    conv1 = fluid.layers.conv2d(input=ipt,
                                num_filters=32, #卷积核的个数
                                filter_size=3, #滤波器大小:高*宽
                                padding=1, #填充大小:高*宽
                                stride=1, #滑动步长：高*宽
                                name='conv1', #网络层输出的前缀标识
                                act='relu') #激活函数。Relu(x)= max(0,x)
    #二维空间池化操作
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2, #池化核的大小
                                pool_stride=2,#池化层的步长
                                pool_type='max',#池化类型
                                name='pool1')
    #批正则化层
    bn1 = fluid.layers.batch_norm(input=pool1, name='bn1')

    conv2 = fluid.layers.conv2d(input=bn1,
                                num_filters=64,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv2',
                                act='relu')

    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool2')

    bn2 = fluid.layers.batch_norm(input=pool2, name='bn2')

    fc1 = fluid.layers.fc(input=bn2, size=1024, act='relu', name='fc1')
    
    # 最终输出10个数
    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')

    return fc2


net = cnn(x)  # CNN模型

cost = fluid.layers.cross_entropy(input=net, label=label) # 交叉熵
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1) # 正确率。如果正确的标签在topk个预测值里，则计算结果+1

# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

# 数据传入设置
#batch_reader = paddle.batch(reader=data_reader(), batch_size=2048)
batch_reader = fluid.io.batch(reader=data_reader(), batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
#trainNum = 15
accL = []
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
        accL.append(outs[2])

    #pross = float(i) / trainNum
    #print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%"+"准确率为"+str(accL[i]))
    if i%5==0:
        print('第'+str(i+1)+'次训练：'+"准确率为"+str(accL[i])+'； 误差为'+str(outs[1]))
        
path = params_dirname
'''
plt.figure(1)
#plt.title('正确率指标')
plt.title('accuracy')
#plt.xlabel('迭代次数')
plt.xlabel('train times')
plt.plot(range(50), accL)
plt.show()
'''
fluid.io.save_inference_model(params_dirname, ['x'], [net], exe)
