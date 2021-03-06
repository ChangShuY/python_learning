'''
来源：
1. PaddlePaddle 极简入门实践四：简单验证码识别
https://www.jianshu.com/p/df98fcc832ed?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=qq
2. PaddlePaddle 极简入门实践八：获取网络中池化\卷积的图像
https://www.jianshu.com/p/9a1628eab8e9

CSY 2020-1-20
改编自：..\test01\code_train.py
可视化损失函数的收敛过程
在行命令窗口执行：
1. 用python运行本程序
   python code_train.py
2. 指定VisualDL的日志文件路径、端口号、IP地址
   visualdl --logdir ../test03/vdl_log --port 8080 --host 127.0.0.1
   或
   visualdl --logdir ./vdl_log --port 8080 --host 127.0.0.1
3. 查看图形。在浏览器的地址栏中输入IP地址和端口号
   http://127.0.0.1:8080

运行环境：Ubuntu-19,python-3.7.0,paddle-1.6.0,visualdl-1.3.0

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


'''
import paddle.fluid as fluid # paddlepaddle 推出的最新的API
#import paddle # 有一些功能是fluid不具备的，还是需要paddle模块
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import visualdl # 如果用 from visualdl import LogWriter , 则下面的 visualdl.LogWriter 可简写为 LogWriter

# 提示 Font family ['sans-serif'] not found.
# 注释掉下面两句
# 绘图部分全部使用英文
#from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置一种用来显示中文的字体


path = "../"
params_dirname = path + "test03/test.inference.model"
print("训练后文件夹路径" + params_dirname)
'''
若不放心当前位置，可以查看当前路径
CSY 2020-1-14
import os
print(os.getcwd())
'''

'''
定义VisualDL日志文件保存在vdl_log文件夹中
第一个参数为日志保存路径，
第二个参数为指定多少次写操作后才从内存写入到硬盘日志文件，越大越占用内存，越小则越占用硬盘IO。
'''
logw = visualdl.LogWriter(path + "test03/vdl_log", sync_cycle=100) #定义保存VisualDL日志文件的路径

# 创建scalar图, mode定义了 line0 显示的名称
with logw.mode('loss') as logger:
    line0 = logger.scalar('train')

# 参数初始化
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 加载数据
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
            #im = im / 255.0

            labelInfo = a[i - 1]
            yield im, labelInfo

    return reader


# 定义网络
#x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype='float32')
x = fluid.data(name="x", shape=[-1, 1, 30, 15], dtype='float32')
#label = fluid.layers.data(name='label', shape=[1], dtype='int64')
label = fluid.data(name='label', shape=[-1,1], dtype='int64')

# CNN,卷积神经网络 Convolutional Neural Network
def cnn(img):
    #第一个卷积-池化层
    '''
    #二维卷积层 输入和输出格式=NCHW 即批尺寸、通道数、特征高度、特征宽度
    conv1 = fluid.layers.conv2d(input=img,
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
    '''
    FilterSize0 = 3 # 第一个卷积层的卷积核大小
    conv_pool_0 = fluid.nets.simple_img_conv_pool(
        input=img, #格式为[N,C,H,W]
        num_filters=32, #卷积核的个数
        filter_size=FilterSize0, #卷积核的大小:高*宽
        pool_size=2, #池化层的大小：高*宽
        pool_stride=2, #池化层的步长
        act='relu' #卷积的激活函数。Relu(x)= max(0,x)
        )
    
    #批正则化层
    bn1 = fluid.layers.batch_norm(input=conv_pool_0, name='bn1')

    #第二个卷积-池化层
    '''
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
    '''
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=bn1, #格式为[N,C,H,W]
        num_filters=50, #卷积核的个数
        filter_size=5, #卷积核的大小:高*宽
        pool_size=2, #池化层的大小：高*宽
        pool_stride=2, #池化层的步长
        act='relu' #卷积的激活函数。Relu(x)= max(0,x)
        )
    
    #批正则化层
    #bn2 = fluid.layers.batch_norm(input=conv_pool_2, name='bn2')
    
    #全连接层
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    
    # 最终输出10个数
    #fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')

    return prediction,conv_pool_0 #返回全连接层和第一个卷积-池化层


net,cp0 = cnn(x)  # CNN模型

cost = fluid.layers.cross_entropy(input=net, label=label) # 交叉熵
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1) # 正确率。如果正确的标签在topk个预测值里，则计算结果+1

# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

# 数据传入设置
#batch_reader = paddle.batch(reader=data_reader(), batch_size=2048)
batch_reader = fluid.io.batch(reader=data_reader(), batch_size=1024)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            #fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
            fetch_list=[label, avg_cost, cp0]) 

    # 打印曲线上的点
    line0.add_record(i,outs[1]) # 横坐标为i,纵坐标为avg_cost
    
    #命令行显示
    #pross = float(i) / trainNum
    #print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%"+"准确率为"+str(accL[i]))
    #if i%5==0:
        #print('第'+str(i+1)+'次训练：'+"准确率为"+str(accL[i])+'； 误差为'+str(outs[1]))
        
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
