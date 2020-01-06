#!/usr/bin/python
#_*_ coding: utf-8 _*_

'''
文件名：Model_aX_Test.py
按照 https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/programming_guide/programming_guide.html#id2
训练的模型，输入x，预测出y_predict。
'''

#加载库
import paddle.fluid as fluid
import numpy

# 定义执行器和作用域
cpu = fluid.CPUPlace()
infer_exe = fluid.Executor(cpu) # 在cpu上运行
inference_scope = fluid.Scope() # 获取作用域变量

# 加载训练好的模型
'''
返回fetch_list=
inference_program -- load_inference_model 返回的预测Program
feed_target_names -- load_inference_model 返回的所有输入变量的名称
fetch_targets     -- load_inference_model 返回的输出变量
参数
params_dirname    -- 待加载模型的存储路径 
infer_exe         -- 运行模型的执行器
'''
params_dirname = 'model'
with fluid.scope_guard(inference_scope):
    [inference_program,feed_target_names,fetch_targets] =  fluid.io.load_inference_model(params_dirname, infer_exe)
#inference_program,feed_target_names,fetch_targets =  fluid.io.load_inference_model(params_dirname, infer_exe)

# 生成测试数据, 模型的输入层设计为只接收float32
group = 4
x_test = numpy.random.uniform(0,10.0,size=(group,1)) #生成8个不同的a, a>=0且a<10, a的类型float64 
x_test = x_test.astype(numpy.float32) # 转换成float32
#x_test = numpy.array([[1.0],[2.0],[3.0],[4.0]]).astype('float32') # x的取值

# 进行预测
results = infer_exe.run(inference_program,
                        feed={'x':x_test},
                        fetch_list=fetch_targets)

# 显示答案
for i in range(group):
    print("当x={x}时，预测结果为y={y}".format(x=x_test[i][0],y=results[0][i]))
