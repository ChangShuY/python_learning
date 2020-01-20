'''
来源：
PaddlePaddle 极简入门实践四：简单验证码识别
https://www.jianshu.com/p/df98fcc832ed?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=qq
学习改编自：predit.py

运行环境：Ubuntu19,python3.7.0,paddle1.6.0
csy 2020-1-16 

加载模型，进行“验证码”预测。

'''

# 加载库
import paddle.fluid as fluid
#import paddle
import numpy as np
#import Class_OS.o1_获得当前工作目录
from PIL import Image

# 指定路径
#path = Class_OS.o1_获得当前工作目录.main()
#params_dirname = path + "test02.inference.CNNmodel"
path = "../"
params_dirname = path + "test01/test.inference.model"
print("训练后文件夹路径" + params_dirname)


def dataReader(i):
    im = Image.open(path + "data/" + str(i) + ".jpg").convert('L')
    im = np.array(im).reshape(1, 1, 30, 15).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


with open(path + "data/ocrData.txt", 'rt') as f:
    a = f.read()

# 参数初始化
#cpu = fluid.CUDAPlace(0)
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
#prog = fluid.default_startup_program()
#exe.run(prog)

# 加载模型
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

#for i in range(1, 2000):
for i in range(1501, 2001):
    img = dataReader(i)
    '''
    results.shape=(1,10)。
    print(results)的显示类似以下内容，
    [array([[4.91851196e-03, 6.52914692e-04, 7.62215466e-04, 6.04133354e-04,
        2.96129001e-04, 2.57041142e-03, 9.84191775e-01, 1.08677756e-04,
        4.81970748e-03, 1.07545953e-03]], dtype=float32)]
   表示：
    0的概率为0.0049
    1的概率为0.00065
    2的概率为0.00076
    3的概率为0.0006
    4的概率为0.000296
    5的概率为0.00257
    6的概率为0.984
    7的概率为0.0001
    8的概率为0.0048
    9的概率为0.00107
    '''
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: img},
                      fetch_list=fetch_targets)
    '''
    np.argsort(results) 对results从小到大排序
    print(np.argsort(results))显示类似以下的内容
    [[[7 4 3 1 2 9 5 8 0 6]]]
    '''
    lab = np.argsort(results)[0][0][-1] # 取概率最大者,[-1]表示最后一个元素，此处等同于[9]
    print(results)
    print(np.argsort(results))
    if str(lab) == a[i - 1]:
        print(str(i)+".jpg：预测结果为" + str(lab) +"与标签一致。结论：正确")
    else:
        print(str(i)+".jpg：预测结果为" + str(lab) +"标签为"+a[i-1]+"结论：错误")
    print()


