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

'''
import paddle.fluid as fluid # paddlepaddle 推出的最新的API
import paddle # 有一些功能是fluid不具备的，还是需要paddle模块
import numpy as np
from PIL import Image # Python Image Libary，是否需要改为pillow
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置一种用来显示中文的字体

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
    # 使用PaddlePaddle中reader生成数据集列表
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
x = fluid.layers.data(name="x", shape=[1, 30, 15], dtype=datatype)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')


def cnn(ipt):
    print(ipt.shape)
    conv1 = fluid.layers.conv2d(input=ipt,
                                num_filters=32,
                                filter_size=3,
                                padding=1,
                                stride=1,
                                name='conv1',
                                act='relu')

    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=2,
                                pool_type='max',
                                name='pool1')

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

    fc2 = fluid.layers.fc(input=fc1, size=10, act='softmax', name='fc2')

    return fc2


net = cnn(x)  # CNN模型

cost = fluid.layers.cross_entropy(input=net, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=net, label=label, k=1)
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
# 数据传入设置
batch_reader = paddle.batch(reader=data_reader(), batch_size=2048)
feeder = fluid.DataFeeder(place=place, feed_list=[x, label])
prog = fluid.default_startup_program()
exe.run(prog)

trainNum = 50
accL = []
for i in range(trainNum):
    for batch_id, data in enumerate(batch_reader()):
        outs = exe.run(
            feed=feeder.feed(data),
            fetch_list=[label, avg_cost, acc])  # feed为数据表 输入数据和标签数据
        accL.append(outs[2])

    pross = float(i) / trainNum
    print("当前训练进度百分比为：" + str(pross * 100)[:3].strip(".") + "%")

path = params_dirname
plt.figure(1)
plt.title('正确率指标')
plt.xlabel('迭代次数')
plt.plot(range(50), accL)
plt.show()

fluid.io.save_inference_model(params_dirname, ['x'], [net], exe)
