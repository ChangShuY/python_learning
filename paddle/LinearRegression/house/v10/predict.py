#引入必要的库
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys
import six

#定义数据提供器
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'convert'
]
feature_num = len(feature_names) #14列
data = numpy.fromfile('housing.data', sep=' ') # 从文件中读取原始数据

#对数据进行预处理
data = data.reshape(data.shape[0] // feature_num, feature_num) # 一大串数据转换为14列一行，抛弃不足一行的数据
#归一化
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
for i in six.moves.range(feature_num-1): # i的取值从0到12，不包含13(最后一列为单价列，不做处理)
    #减去平均值，除以取值范围
   data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i]) # six.moves可以兼容python2和python3

ratio = 0.8 # 训练集和验证集的划分比例
offset = int(data.shape[0]*ratio)
train_data = data[:offset]
test_data = data[offset:]

def reader_creator(train_data):  
    def reader():  
        for d in train_data:
            # yield 用法见 https://blog.csdn.net/mieleizhi0522/article/details/82142856
            # d每次取train_data的一行
            # 转换成两个一维张量。data[:-1]得到该行的去掉后尾的数据；d[-1:]取最后一个数
            yield d[:-1], d[-1:]  
    return reader

BATCH_SIZE = 20

test_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(test_data), buf_size=500),
        batch_size=BATCH_SIZE)

#定义运算场所
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() # 指明executor的执行场所

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')

params_dirname = "fit_a_line.inference.model"

with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe) # 载入预训练模型
    batch_size = 10

    infer_reader = paddle.batch(
        paddle.dataset.uci_housing.test(), batch_size=batch_size) # 准备测试集

    infer_data = next(infer_reader())
    infer_feat = numpy.array(
        [data[0] for data in infer_data]).astype("float32") # 提取测试集中的数据
    infer_label = numpy.array(
        [data[1] for data in infer_data]).astype("float32") # 提取测试集中的标签

    assert feed_target_names[0] == 'x'
    results = infer_exe.run(inference_program,
                            feed={feed_target_names[0]: numpy.array(infer_feat)},
                            fetch_list=fetch_targets) # 进行预测
    #打印预测结果和标签并可视化结果
    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val)) # 打印预测结果

    print("\nground truth:")
    for idx, val in enumerate(infer_label):
        print("%d: %.2f" % (idx, val)) # 打印标签值

    save_result(results[0], infer_label) # 保存图片
