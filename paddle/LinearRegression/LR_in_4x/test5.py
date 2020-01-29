'''
运行环境：Ubuntu19,python3.7.0,paddle1.6.0
csy 2020-1-10 源于
https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html

1.fluid.layers.data 改为 fluid.layers.data
2.测试数据的shape改为(1,4)
3.生成的训练数据改为30组
4.生成的x小于10。
5.增加打印训练数据x,y的语句
运行结果：
0: 47=4*5+6*0+7*3+2*3
1: 113=4*7+6*9+7*3+2*5
2: 93=4*2+6*4+7*7+2*6
3: 99=4*8+6*8+7*1+2*6
4: 128=4*7+6*7+7*8+2*1
5: 148=4*5+6*9+7*8+2*9
6: 40=4*4+6*3+7*0+2*3
7: 40=4*5+6*0+7*2+2*3
8: 65=4*8+6*1+7*3+2*3
9: 56=4*3+6*7+7*0+2*1
10: 98=4*9+6*9+7*0+2*4
11: 74=4*7+6*3+7*2+2*7
12: 16=4*2+6*0+7*0+2*4
13: 108=4*5+6*5+7*6+2*8
14: 68=4*4+6*1+7*4+2*9
15: 59=4*8+6*1+7*1+2*7
16: 123=4*9+6*9+7*3+2*6
17: 46=4*7+6*2+7*0+2*3
18: 110=4*5+6*9+7*4+2*4
19: 82=4*6+6*4+7*4+2*3
20: 104=4*4+6*4+7*8+2*4
21: 99=4*3+6*7+7*5+2*5
22: 59=4*0+6*1+7*5+2*9
23: 47=4*3+6*0+7*5+2*0
24: 48=4*1+6*2+7*4+2*2
25: 32=4*0+6*3+7*2+2*0
26: 121=4*7+6*5+7*9+2*0
27: 82=4*2+6*7+7*2+2*9
28: 51=4*2+6*3+7*3+2*2
29: 47=4*3+6*4+7*1+2*2
iter=0,cost=5474.8759765625
iter=50,cost=nan
iter=100,cost=nan
iter=150,cost=nan
iter=200,cost=nan
iter=250,cost=nan
iter=300,cost=nan
iter=350,cost=nan
iter=400,cost=nan
iter=450,cost=nan
9a+5b+2c+10d=[nan]
'''
#加载库
import paddle.fluid as fluid
import numpy as np
#生成数据
np.random.seed(0)
outputs = np.random.randint(10, size=(30, 4))
res = []
for i in range(30):
        # 假设方程式为 y=4a+6b+7c+2d
        y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
        print("{}: {}=4*{}+6*{}+7*{}+2*{}".format(i,y,outputs[i][0],outputs[i][1],outputs[i][2],outputs[i][3]))
        res.append([y])
# 定义数据
train_data=np.array(outputs).astype('float32')
y_true = np.array(res).astype('float32')

#定义网络
x = fluid.data(name="x",shape=[-1,4],dtype='float32')
y = fluid.data(name="y",shape=[-1,1],dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)
#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)
#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
sgd_optimizer.minimize(avg_cost)
#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())
##开始训练，迭代500次
for i in range(500):
        outs = exe.run(
                feed={'x':train_data,'y':y_true},
                fetch_list=[y_predict.name,avg_cost.name])
        if i%50==0:
                print ('iter={:.0f},cost={}'.format(i,outs[1][0]))
#存储训练结果
params_dirname = "result"
fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

# 开始预测
infer_exe = fluid.Executor(cpu)
inference_scope = fluid.Scope()
# 加载训练好的模型
with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)

# 生成测试数据
test = np.array([[9,5,2,10]]).astype('float32')
# 进行预测
results = infer_exe.run(inference_program,
                                                feed={"x": test},
                                                fetch_list=fetch_targets)
# 给出题目为 【9,5,2,10】 输出y=4*9+6*5+7*2+10*2的值
print ("9a+5b+2c+10d={}".format(results[0][0]))
