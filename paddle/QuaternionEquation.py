##一个简单的线性回归模型，来帮助我们快速求解4元一次方程。
## https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/quick_start_cn.html

#加载库
import paddle.fluid as fluid
import numpy as np

#准备训练数据
np.random.seed(0)
#生成10行4列，取值小于5的随机整数
outputs = np.random.randint(5, size=(10, 4))
res = [] #定义1个空张量
for i in range(10):
        # 假设方程式为 y=4a+6b+7c+2d
        # 将随机数据作为自变量，求出因变量作为标准答案保存
        y = 4*outputs[i][0]+6*outputs[i][1]+7*outputs[i][2]+2*outputs[i][3]
        res.append([y])

# 定义数据,同时将整数张量转换为浮点数张量
train_data=np.array(outputs).astype('float32')
y_true = np.array(res).astype('float32')

#定义网络
x = fluid.layers.data(name="x",shape=[4],dtype='float32')
y = fluid.layers.data(name="y",shape=[1],dtype='float32')
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
        #outs[0]存放10个预测值，outs[1]存放1个误差值
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
test = np.array([[[9],[5],[2],[10]]]).astype('float32')
# 进行预测
results = infer_exe.run(inference_program,
                        feed={"x": test},
                        fetch_list=fetch_targets) 

# 给出题目为 【9,5,2,10】 输出y=4*9+6*5+7*2+10*2的值
print ("9a+5b+2c+10d={}".format(results[0][0]))
