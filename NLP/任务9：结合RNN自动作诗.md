[TOC]
# 任务9：结合RNN自动作诗

## 1 任务目标

1. 掌握循环神经网络的结构和原理
2. 学会使用循环神经网络生成文本（基于TensorFlow框架）

## 2 任务描述

在一段文本中，词与词，句子与句子之间有很强的关联性。那么如何从已经观测到的字符出发，预测下一个字符出现的概率呢？

本章我们就来学习一下非常擅长处理序列问题的循环神经网络（RNN）。

自然语言和音频都是前后相互关联的数据，对于这些序列数据需要使用 循环神经网络(Recurrent Neural Network，RNN) 来进行处理。

使用循环结构拥有很多优势，最突出的一个优势就是它们能够在内存中存储前一个输入的表示。如此，我们就可以更好的预测后续的输出内容。持续追踪内存中的长数据流会出现很多的问题。

循环神经网络已经被用于语音识别、语言翻译、股票预测等等，它甚至用于图像识别来描述图片中的内容。

那么我们就来学习一下RNN的结构原理，以及如何训练自己的RNN文本生成模型，实现AI作诗吧。



## 3 知识准备


### 3.1 什么是序列数据？
假设你拍摄了一张球在时间上移动的静态快照：

![image](https://pic4.zhimg.com/80/v2-50b70da4c5c3116f5d29a7aecedf8af3_hd.jpg)

如果我们想要预测这个球的移动方向，显然这一张照片是不够的。

所以我们连续记录球位置的快照：

![image](https://pic3.zhimg.com/v2-8eae31d397dbdd19aa7b57bd375403c6_b.webp)

这样一来，我们就能有足够的信息来做出更好的预测了。

这就是一个序列：一个事物按照一个特定的顺序跟随着另一个事物。

序列数据有很多种形式。我们来看一些生活中的例子：

![image](https://raw.githubusercontent.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai/master/05-Sequence%20Models/week1/md_images/01.png)

- 语音识别：将输入的语音信号*x*直接输出相应的语音文本信息*y*。无论是语音信号还是文本信息均是**序列数据**。
- 音乐生成：生成音乐乐谱。此时只有输出的音乐乐谱*y*是**序列数据**，而输入*x*可以为空或者是一个整数。
- 情感分类：将输入的评论句子*x*转换成相应的等级或者是评分，此时输入是一个**序列数据**，输出则是一个单独的类别。
- DNA序列分析：输入的是**DNA序列**，输出的是蛋白质表达的**子序列**。
- 机器翻译：两种不同语言之间的相互转换。输入和输出都是**文本序列**。
- 食品行为识别：输入的是**视频帧序列**，输出的是动作分类。
- 命名实体识别：输入的是**句子序列**，输出的是实体名字。 

从这一系列例子中我们可以看出序列问题其实有很多不同的类型。

有些问题里，输入和输出的数据都是序列，而且就算是那种情况下，输入和输出的序列也不一定会一样长。

而另外一些问题里，只有输入或者输出才是序列数据。

### 3.2 如何用数学符号表示文本序列？

经过前面的学习我们知道，想要表示一个词语，需要先建立一个 词汇表(Vocabulary) ，或者叫字典(dictionary) 。将需要表示的所有词语变为一个列向量，可以根据字母顺序排列，然后根据单词在向量中的位置，用 one-hot 向量(one-hot vector) 来表示该单词的标签。



### 3.3 循环神经网络模型

#### 3.3.1 为什么不继续使用普通卷积网络

现在假设我们有一个句子：“张三和李四是好朋友。”

传统的神经网络结构图如下所示：

![image](https://pic4.zhimg.com/80/v2-274ca6499a13292208fff69e8e3fadaf_hd.jpg)

如果我们按照上一个任务的做法，把这九个汉字用九个one-hot向量表示，然后将他们输入到标准的神经网络中之中，经过一些隐藏层，最后会输出九个值为0或者为1的项，表明每个输入汉字是否是人名的一部分。

具体来说，如果神经网络已经学习到出现在位置1的“张”是人名的一部分，那么当“张”出现在其他位置的时候，它也能识别“张”很有可能是人名的一部分的话，这就达到了很不错的效果。

但是只是一个单纯的神经网络结构，它并不能共享从文本的不同位置上学习到的特征。

#### 3.3.2 循环神经网络

那么我们如何在传统神经网络的基础上加入循环结构，以便能够使用以前的信息来影响以后的信息呢？

我们考虑在神经网络中添加一个可以传递先前信息的循环：

![image](https://pic4.zhimg.com/v2-124f53ff4804c87b62c0d0aeea3536d3_b.webp)

循环输入的信息是隐藏状态，它是先前输入的表示。

#### 3.3.3 实例说明循环神经网络

我们知道聊天机器人可以根据用户输入的文本对意图进行分类。

![image](https://pic4.zhimg.com/v2-0e23dc6be92a34fc2fd7bdc75df653f7_b.jpg)

为了实现一个聊天机器人，我们的思路是：

- 首先，我们将使用RNN对文本序列进行编码。
- 然后，我们将RNN输出馈送到前馈神经网络中，该网络将对用户输入意图进行分类。

假设用户输入：what time is it？

首先，我们将句子分解为单个单词。

RNN按顺序工作，所以我们一次只能输入一个字。

1. 我们第一步是把一个句子分成单词序列：

![image](https://pic1.zhimg.com/v2-a193b795d14b43c0ad4c4807f28d7204_b.webp)

2. 接着我们把“what”输入RNN，RNN对其编码并产生输出：

![image](https://pic1.zhimg.com/v2-16e09dbc9c67724fb76ede680df31f14_b.webp)

3. 我们提供单词“time”和上一步中的隐藏状态。RNN现在有关于“what”和“time”这两个词的信息。

![image](https://pic1.zhimg.com/v2-93df2c57a2c7556a022708eab178f740_b.webp)

4. 重复这个过程，直到最后一步。你可以通过最后一步看到RNN编码了前面步骤中所有单词的信息。

![image](https://pic3.zhimg.com/v2-d54f07f446620ea7f294953b226db702_b.jpg)
5. 由于最终输出是从序列的部分创建的，因此我们应该能够获取最终输出并将其传递给前馈层以对意图进行分类。

![image](https://pic3.zhimg.com/v2-7c62e9d3c31f4d755bc73f1849a53532_b.jpg)

整个过程用伪代码表示如下：

```python
# 初始化网络层和初始隐藏状态
rnn = RNN()
ff = FeedForwardNN()

# 定义隐藏状态的形状和维度
hidden_state = [0.0, .0., 0.0, 0.0]

# 循环输入，将单词和隐藏状态传递给RNN。RNN返回输出和修改的隐藏状态
for word in input:
    output, hidden_state = rnn(word, hidden_state)
    
# 将输出传递给前馈层，然后返回预测
prediction = ff(output)
```

### 3.4 LSTM和GRU
RNN会受到短期记忆的影响，那么我们如何应对呢？

为了减轻短期记忆的影响，研究者们创建了两个专门的递归神经网络，一种叫做长短期记忆或简称LSTM。另一个是门控循环单位或GRU。

LSTM和GRU本质上就像RNN一样，但它们能够使用称为“门”的机制来学习长期依赖。

这些门是不同的张量操作，可以学习添加或删除隐藏状态的信息。由于这种能力，短期记忆对他们来说不是一个问题。

总的来说，RNN具有更快训练和使用更少计算资源的优势，这是因为要计算的张量操作较少。当你期望对具有长期依赖的较长序列建模时，你应该使用LSTM或GRU。
## 4 任务实施

### 4.1 实施思路
上面讲了这么多理论细节，那么如何完成本节课的任务呢？

1. 准备数据集
2. 数据预处理
2. 配置RNN参数
3. 实现RNN
6. 训练模型
7. 测试模型

#### 步骤1：准备数据集

本任务提供了英文、唐诗、周杰伦歌词、莎士比亚小说、代码、日文等数据集，读者可以任选其中一种来进行模型训练。

本次任务使用了3w字的唐诗数据集来训练自动写诗模型。

#### 步骤2：数据预处理

本步骤定义的类主要用于将文本转化成数值存入矩阵，或将数值转回文本。

```
import numpy as np
import copy
import pickle

# 生成每批次喂入网络训练的样本
def batch_generator(arr, n_seqs, n_steps):
    """
    :param arr: 训练集数据
    :param n_seqs:一个batch的句子数量，32
    :param n_steps: 句子长度，26
    :return: x, y 的生成器
    """
    # 把数据备份一份
    # copy.copy是一种对象复制方法，也叫浅复制，修改子对象会对其造成影响
    arr = copy.copy(arr)  
    
    # 一个batch的句子数量*句子长度=一个batch的总字数
    batch_size = n_seqs * n_steps  
    
    n_batches = int(len(arr) / batch_size)  # 取到了batch的整数
    arr = arr[:batch_size * n_batches]  # [:n_seqs * n_steps * n_batches]
    arr = arr.reshape((n_seqs, -1))  # # [n_seqs: n_steps * n_batches]
    while True:
    
        # 打乱矩阵内部元素
        np.random.shuffle(arr)
        
        # 每次循环是一次batch
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]  # 一个句子，句子的每个词
            y = np.zeros_like(x) # 用于生成大小一致的tensor但所有元素全为0
            
            # y[:, -1]所有行的最后一列=x[:, 0] 所有行的第0列
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            
            # yield的使用是将函数作为生成器，这样做省内存
            yield x, y


class TextConverter(object):

    # text表示待转化的文本，直接通过参数的形式传入
    # max_vocab表示字符的最大容量
    # filename表示带转化的文档
    
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)  # 变成集合，去重
            # print("数据集总共用到了多少词", len(vocab))  
            # max_vocab_process
            # 计算每个词出现的次数
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1

            vocab_count_list = []  # [(词，词数量), (词，词数量)...]
            for word in vocab_count:  # 字典循环，得到的是键
                vocab_count_list.append((word, vocab_count[word]))
            # 按照词数量倒序 大-->小
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)  
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab  # 装载所有词的列表

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        
        # {(索引，单词),(索引，单词)...}
        self.int_to_word_table = dict(enumerate(self.vocab)) 
        
        # 遍历字典中的元素
        for item in list(self.int_to_word_table.items())[:50]:  
            print(item)
            # (0, '，')
            # (1, '。')
            # (2, 'n')
            # (3, '不')
            # (4, '人')
            # (5, '山')
            # (6, '风')
            # (7, '日')
            # (8, '云')
            # (9, '无')
            # (10, '何')
            # (11, '一')
            # (12, '春')
            # (13, '月')
            # (14, '水')
            # (15, '花')

    @property
    # 返回字符的数量+1
    def vocab_size(self):
        return len(self.vocab) + 1

    # 如果字符有索引则返回索引，没有则返回最后那个数值
    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]  # 返回这是第几个词
        else:
            return len(self.vocab)

    # 如果索引是最后那个数值，则返回表示unknown
    # 其他索引则返回相应字符，另外超出范围的索引则抛出异常
    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]  # 返回第几个词所对应的词
        else:
            raise Exception('Unknown index!')

    # 文字逐一转化为索引后生成矩阵
    def text_to_arr(self, text):
        arr = []
        
        # text中的词，出现在vocab中的索引
        for word in text:
            arr.append(self.word_to_int(word))  
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
```

#### 步骤3：配置RNN参数

通过tf.flags.DEFINE_string定义一些全局参数：

```
import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', '模型名')
tf.flags.DEFINE_integer('num_seqs', 32, '一个batch里面的序列数量')       # 32
tf.flags.DEFINE_integer('num_steps', 26, '序列的长度')                   # 26
tf.flags.DEFINE_integer('lstm_size', 128, 'LSTM隐层的大小')
tf.flags.DEFINE_integer('num_layers', 2, 'LSTM的层数')
tf.flags.DEFINE_boolean('use_embedding', False, '是否使用 embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding的大小')
tf.flags.DEFINE_float('learning_rate', 0.001, '学习率')
tf.flags.DEFINE_float('train_keep_prob', 0.5, '训练期间的dropout比率')
tf.flags.DEFINE_string('input_file', '', 'utf8编码过的text文件')
tf.flags.DEFINE_integer('max_steps', 10000, '一个step 是运行一个batch， max_steps固定了最大的运行步数')
tf.flags.DEFINE_integer('save_every_n', 1000, '每隔1000步会将模型保存下来')
tf.flags.DEFINE_integer('log_every_n', 10, '每隔10步会在屏幕上打出曰志')
# 使用的字母（汉字）的最大个数。默认为3500 。程序会自动挑选出使用最多的字，井将剩下的字归为一类，并标记为＜unk＞
tf.flags.DEFINE_integer('max_vocab', 10000, '最大字符数量')

```
#### 步骤4：实现RNN
```
import os
import time
import numpy as np
import tensorflow as tf


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # p[np.argsort(p)]将p从小到大排序
    p[np.argsort(p)[:-top_n]] = 0  # 将除了top_n个预测值的位置都置为0
    p = p / np.sum(p)  # 归一化概率
    # 以p的概率从vocab_size中随机选取一个字符，p是列表,vocab_size也是列表，p代表vocab_size中每个字的概率
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class CharRNN:
    def __init__(self, num_classes, num_seqs=32, num_steps=26, lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        if sampling is True:  # 如果是测试
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes  # 一共分3501类，每个字是一类，判断下一个字出现的概率，是下一个类的概率，分类任务
        self.num_seqs = num_seqs  # 一个batch里面句子的数量32
        self.num_steps = num_steps  # 句子的长度26
        self.lstm_size = lstm_size  # 隐藏层大小 （batch_size, state_size）
        self.num_layers = num_layers  # LSTM层数量
        self.learning_rate = learning_rate  # 学习率
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size  # embedding的大小128

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            # shape = (batch_size, num_steps) = (句子数量，句子长度)=(32, 26)
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            # 输出shape=输入shape，内容是self.inputs每个字母对应的下一个字母(32, 26)
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # 对于汉字生成，使用embedding层会取得更好的效果。
            # 英文字母没有必要用embedding层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    # 先定义一个embedding变量，embedding才是我们的训练数据(字的总类别，每个字的向量)=(3501, 128)
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    # 使用tf.nn.embedding lookup查找embedding，让self.input从embedding中查数据
                    # 请注意embedding变量也是可以训练的，因此是通过训练得到embedding的具体数值。

                    # embedding.shape=[self.num_classes, self.embedding_size]=(3501, 128)
                    # self.inputs.shape=(num_seqs, num_steps)=(句子数量，句子长度)=(32, 26)
                    # self.lstm_inputs是直接输入LSTM的数据。
                    # self.lstm_inputs.shape=(batch_size, time_step, input_size)=(num_seqs, num_steps, embedding_size)=(句子数量，句子长度，词向量)=(32, 26, 128)
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        """定义多层N vs N LSTM模型"""

        # 创建单个cell函数
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        # 将LSTMCell进行堆叠
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])
            # 隐藏层的初始化 shape=batch_size，计入笔记中，你的博客漏掉了
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)     # (batch_size, state_size)
            print("self.initial_state.shape", self.initial_state)
            # (LSTMStateTuple(
            #   c= <tf.Tensor 'lstm/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros:0' shape = (32, 128) dtype = float32 >,
            #   h = < tf.Tensor 'lstm/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1:0' shape = (32, 128) dtype = float32 >),
            # LSTMStateTuple(
            #   c= < tf.Tensor 'lstm/MultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros:0' shape = (32, 128) dtype = float32 >,
            #   h = < tf.Tensor 'lstm/MultiRNNCellZeroState/DropoutWrapperZeroState_1/BasicLSTMCellZeroState/zeros_1:0' shape = (32, 128) dtype = float32 >))

            # 将我们创建的LSTMCell通过dynamic_rnn对cell展开时间维度，不然只是在时间上走"一步"
            # inputs_shape = (batch_size, time_steps, input_size)
            # initial_state_shape = (batch_size, cell.state_size)
            # output_shape=(batch_size, time_steps, cell.output_size)=(32, 26, 128) time_steps步里所有输出，是个列表
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)
            # 通过lstm_outputs得到概率
            seq_output = tf.concat(self.lstm_outputs, 1)  # 合并所有time_step得到输出,lstm_outputs只有一个，因此还是原shape=32, 26, 128)
            x = tf.reshape(seq_output, [-1, self.lstm_size])    # (batch_size*time_steps, cell.output_size)=(32*26, 128)

            # softmax层
            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b  # 预测值
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')  # 变成下一个词出现的概率

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        # 使用截断梯度下降 clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, self.final_state, self.optimizer], feed_dict=feed)

                end = time.time()
                # control the print lines
                if step % log_every_n == 0:
                    print('step: {}/{}... '.format(step, max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if step % save_every_n == 0:
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def sample(self, n_samples, prime, vocab_size):
        """
        :param n_samples: 生成多少词
        :param prime:       开始字符串
        :param vocab_size: 一共有多少字符
        """
        samples = [c for c in prime]  # [6, 14]=[风, 水]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size,))  # for prime=[]
        for c in prime:
            print("输入的单词是：", c)
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            # preds是概率，
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        print("预测出的词是", c)      # 18-->中
        samples.append(c)   # 添加字符到samples中

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):  # 30
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

            c = pick_top_n(preds, vocab_size)       # c 为词索引
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
```

#### 步骤5：训练模型

训练的主函数。

os.path.join生成model的路径名，如果model这个文件夹不存在，则创建一个文件夹；

读取文件后，创建一个converter对象（TextConverter类），arr储存文本转数值的矩阵，g用来存储batch（包括训练输入x及标签y）。

创建一个model对象（CharRNN类），并进行训练。

tf.app.run()表示运行程序后执行main函数。

```
def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:  # 打开训练数据集poetry.txt
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)    # 最大字符数量10000
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))

    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)       # 句子数量、句子长度
    print(converter.vocab_size)     # 3501
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
    model.train(g, FLAGS.max_steps, model_path, FLAGS.save_every_n, FLAGS.log_every_n)


if __name__ == '__main__':
    tf.app.run()
```
训练过程：
![image](https://s2.ax1x.com/2019/10/17/KE8UaQ.png)

#### 步骤6：测试模型

首先也是创建了一个converter对象；然后找到checkpoint，即之前训练好的模型文件；创建model对象，载入这个checkpoint；将start从字符串转为数值矩阵，通过sample方法开始生成；最后将结果转化为文本打印出来。

```
import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')
# --use_embedding --start_string "风水" --converter_path model/poetry/converter.pkl --checkpoint_path model/poetry/ --max_length 30


def main(_):
    FLAGS.start_string = FLAGS.start_string
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size,
                    sampling=True,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)
    print("arr装的是每个单词的位置", arr)
    print(converter.arr_to_text(arr))


if __name__ == '__main__':
    tf.app.run()
```
模型输出结果：

![image](https://s2.ax1x.com/2019/10/17/KEbhbn.png)


## 5 任务拓展
**梯度消失：**

RNN存在的短期记忆问题是由臭名昭着的梯度消失问题引起的，这在其他神经网络架构中也很普遍。

由于RNN处理很多步骤，因此难以保留先前步骤中的信息。

正如你所看到的，在最后的时间步骤中，“what”和“time”这个词的信息几乎不存在。

短期记忆和梯度消失是由于反向传播的性质引起的，反向传播是用于训练和优化神经网络的算法。

训练神经网络有三个主要步骤。

首先，它进行前向传递并进行预测。

其次，它使用损失函数将预测与基础事实进行比较。

损失函数输出一个错误值，该错误值是对网络执行得有多糟糕的估计。

最后，它使用该误差值进行反向传播，计算网络中每个节点的梯度。

梯度是用于调整网络内部权重的值从而更新整个网络。

梯度越大，调整越大，反之亦然，这也就是问题所在。

在进行反向传播时，图层中的每个节点都会根据渐变效果计算它在其前面的图层中的渐变。

因此，如果在它之前对层的调整很小，那么对当前层的调整将更小。
这会导致渐变在向后传播时呈指数级收缩。

由于梯度极小，内部权重几乎没有调整，因此较早的层无法进行任何学习。这就是消失的梯度问题。

由于梯度消失，RNN不会跨时间步骤学习远程依赖性。

这意味着在尝试预测用户的意图时，有可能不考虑“what”和“time”这两个词。

然后网络就可能作出的猜测是“is it？”。

这很模糊，即使是人类也很难辨认这到底是什么意思。

因此，无法在较早的时间步骤上学习会导致网络具有短期记忆。
## 6 任务实训


### 6.1 实训目的
1. 掌握循环神经网络的结构和原理
2. 学会使用循环神经网络训练文本（基于TensorFlow框架）

### 6.2 实训内容
- 动手修改RNN配置中的参数，更换不同的数据集，重新训练属于你自己的模型，对比一下不同参数设置会对训练出来的模型的准确度产生什么影响吧！

