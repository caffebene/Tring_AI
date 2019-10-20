[TOC]
# 任务10：长短期记忆网络LSTM

## 1 任务目标

- 了解长短期记忆网络LSTM和循环神经网络的主要区别

- 理解长短期记忆网络LSTM的网络结构

- 学习搭建一个简单的长短期记忆网络LSTM

  


## 2 任务描述

- 在TensorFlow深度学习框架下，搭建一个简单的长短期记忆网络LSTM

- 使用长短期记忆网络LSTM完成手写数字识别任务

  


## 3 知识准备

### 3.1 长短期记忆网络LSTM

- 长短期记忆网络(Long Short-Term Memory，LSTM)，是循环神经网络RNN的一种变体。LSTM模型设计之初是为了解决循环神经网络中长期依赖的问题，也就是循环神经网络无法保留太久远以前的信息。而LSTM模型可以记住很长时间内的信息，而且很轻松就能做到。
- 下一节我们会给出LSTM的模型结构，从模型输入输出来看，长短期记忆网络和循环神经网络是一致的，因此从实际应用的角度上，循环神经网络RNN能够实现的任务如自然语言处理、时间序列预测等，长短期记忆网络LSTM都可以实现，而且由于大部分任务都需要依赖于长期信息才能做出预测，因此长短期记忆网络的表现往往比循环神经网络更好。



### 3.2 LSTM模型结构

![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/rnn.png)
																				**RNN网络结构示意图**

![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/lstm.png)
																				**LSTM网络结构示意图**

- 从上面两幅图我们可以看出，长短期记忆网络LSTM对循环神经网络RNN的改进主要在于内部的运算不同，新增加了三种运算（或者直观上的解释是增加了三种门结构），同时多了一个状态变量$C$，也正因为如此，LSTM弄够更好的保留长期的信息。下面我们将详细解释LSTM中新增的三种门结构
- 遗忘门，Forget Gate Layer

![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/forget.png)

​		第一个门结构是遗忘门，作用是将$h_{t-1}和x_t$的一部分不重要的信息遗忘，而重要的信息通过$C$保留给下一个时刻。

- 输入门，Input Gate Layer

  ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/input1.png)

  ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/input2.png)

  第二个门结构是输入门，输入门分为两步，第一步是通过$\sigma$函数决定$\tilde{C}_t $哪些信息需要更新，而$\tilde{C}_t $则和原来RNN的计算方式一样，是通过一个非线性函数tanh计算而来。

  输入门的第二步是利用$f_t$来决定$C_{t-1}$中哪些信息需要遗忘掉，同时将第一步得到的关于$h_{t-1}和x_t$要保留的信息保留在$C_t$之中。由此得到了t时刻的状态$C_t$

- 输出门，Output Gate Layer

    ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/output.png)

  第三个门结构是输出门，这时的输出$h_t$根据状态$C_t$进行计算，然后根据$o_t$选择$C_t$中要用到的信息，得到最终神经元的输出。



## 4 任务实施

### 4.1 实施思路

- 这一节中我们将在TensorFlow框架下搭建一个长短期记忆网络LSTM并对手写数字进行识别，实施思路如下。
  1. 加载公开手写数字数据集MNIST，并转换成网络输入要求的格式
  2. 使用TensorFlow搭建一个长短期记忆网络LSTM
  3. 使用第1步中的数据作为训练集，训练LSTM网络中的参数
  5. 使用第3步训练后得到的LSTM网络进行预测，观察长短期记忆网络的准确率。




### 4.2 实施步骤

#### 步骤1：加载MNIST数据集

```python
# 导入数据，Tensorflow库中带有MNIST这个数据集
mnist = input_data.read_data_sets('./mnist', one_hot=True)     # 以one_hot方式编码读入数据集,其中图片中的像素值已经做了归一化处理
test_x = mnist.test.images[:2000]   #取测试集的前2000张作为测试
test_y = mnist.test.labels[:2000]
```



#### 步骤2：搭建长短期记忆网络LSTM

```python
# 定义LSTM的输入输出
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # 输入的维度是(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # 转换成LSTM要求的维度
tf_y = tf.placeholder(tf.int32, [None, 10])                             # 输出维度是(batch,10)


#搭建一个LSTM模型，Tensorflow中已经有封装好的函数
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=64,reuse=tf.AUTO_REUSE) #设置LSTM中节点数为64
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    lstm_cell,                   # 上面搭建的LSTMcell
    image,                      #  输入
    initial_state=None,         # 初始隐状态
    dtype=tf.float32,           #  如果 initial_state = None，需要指定数据类型
    time_major=False,           # False时: (batch, time step, input); True时: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 10)              # 将上面得到的outputs与十个节点进行全连接，十个节点分布表示0-9


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)        # 以交叉熵作为预测的损失函数
train_op = tf.train.AdamOptimizer(LR).minimize(loss)  #选择Adam优化器作为优化方法
```



#### 步骤3：训练LSTM

```python
accuracy = tf.metrics.accuracy(          # 利用tf里的api计算准确率
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # 初始化所有变量
sess.run(init_op)     # 使上一句初始化语句在计算图中运行

for step in range(150):    # 训练150次
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 15 == 0:      # 每训练15次在测试集上测试LSTM的准确率
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('训练误差: %.4f' % loss_, '| 测试准确率: %.2f' % accuracy_)
```




#### 步骤4：使用LSTM进行手写数字预测

```python
test_output = sess.run(output, {tf_x: test_x[:10]}) #对测试集上的前10张图片进行预测
pred_y = np.argmax(test_output, 1)  #test_output是一个list，np.argmax取list中最大数字的下标即预测值。
print(pred_y, '预测值')
print(np.argmax(test_y[:10], 1), '实际值')
```




## 5 任务拓展

- 上面介绍的是非常一般的长短期记忆网络，而如今LSTM模型也产生了许多变种，主要是对LSTM模型中的几种门结构进行分析，得到更精简、模型参数量更少的神经网络，其中比较有效的几个变种是Peephole LSTM以及GRU模型。

- Peephole LSTM, 与原来的LSTM模型结构进行对比就可以发现。该改进主要是通过$\sigma$函数决定要保留$h_{t-1}和x_t$中哪些信息时，引入$C_{t-1}$或者$C_t$的信息再进行决定，由于添加了更多有效的信息来进行判断，$\sigma$ 函数的决定也会更正确。

  ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/peepholes.png)

- GRU模型，全称是Gated Recurrent Unit，该改进主要是将状态$C和h$进行了合并，然后将遗忘门结构和输入门结构通过$'1-'$这个操作进行耦合，使两个门结构巧妙地互相牵扯，使得模型的参数量更少更简单，具体细节可以查阅更多信息进行了解。

  ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/GRU.png)

## 6 任务实训

### 6.1 实训目的

- 自主学习如何使用Tensorflow做数据格式转换
- 了解长短期记忆网络LSTM的输入格式要求

### 6.2 实训内容

- 在提供的Jupyter notebook样例中，尝试将测试数据替换成下面的图片或者一张网上的手写数字图片，并观测网络预测结果。

  ![](https://github.com/caffebene/Tring_AI/raw/master/Deep_Learning/RNN/CH2_LSTM/9.jpg)


