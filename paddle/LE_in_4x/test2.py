'''
运行环境：Ubuntu19,python3.7.0,paddle1.6.0
csy 2020-1-10 源于
https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html

1.fluid.layers.data 改为 fluid.layers.data
2.测试数据的shape改为(1,4)
3.生成的训练数据改为20组
运行结果：
iter=0,cost=1878.2529296875
iter=50,cost=6270566.5
iter=100,cost=21211248640.0
iter=150,cost=71750607962112.0
iter=200,cost=2.4270867061643674e+17
iter=250,cost=8.210029047386651e+20
iter=300,cost=2.7771812434181936e+24
iter=350,cost=9.394261493362e+27
iter=400,cost=3.1777560548561466e+31
iter=450,cost=1.0749267084194374e+35
9a+5b+2c+10d=[-5.5128814e+19]

'''

#加载库
import paddle.fluid as fluid
import numpy as np
#生成数据
np.random.seed(0)
outputs = np.random.randint(5, size=(20, 4))
res = []
for i in range(20):
        # 假设方程式为 y=4a+6b+7c+2d
        y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
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
