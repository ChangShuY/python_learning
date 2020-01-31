'''
CSY 2020-1-31
收录在博客：https://blog.csdn.net/aLife2P6/article/details/104108872
'''
import paddle.fluid as fluid
import numpy

#train_data=numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32') # 食物的shape=(4,1)，type='float32'
def reader():
    for i in range(1,11):
        train_data = numpy.array([i,i*2,i*3,i*4]).reshape(4, 1).astype("float32")  
        yield train_data    # 使用yield来返回单条数据

x = fluid.data(name="x",shape=[None,1],dtype='float32') # 嘴巴的shape容纳了食物的shape。

y_predict = fluid.layers.fc(input=x,size=1,act=None)

cpu = fluid.core.CPUPlace() # 在cpu上操作
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program()) # 运行执行器初始化网络参数（采用默认的初始化程序）

for i in range(3): # 连吃三顿
    print(i) # i表示第几顿
    m=0 # m表示第几道
    for data in reader():  #每顿吃10道“菜”，每口只喂1道“菜”
        if(i==2):
            print(m,data) #显示最后一顿的菜谱
        m+=1
        outs = exe.run( # 加载主程序运行执行器
        feed={'x':data}, # 从名为x的嘴巴喂入reader端上的食物
        fetch_list=[y_predict])
# 最后的训练结果
print(outs) # 输出列表仅有一个内容，就是out[0]=y_predict