[TOC]
# 任务8：结合CNN实现文本分类

## 1 任务目标

1. 掌握卷积神经网络的结构和原理
2. 熟练使用Word2vec训练文本生成模型
2. 学会使用卷积神经网络训练文本（基于TensorFlow框架）

## 2 任务描述

NLP 里面，传统的句子分类器一般使用支持向量机分类器（SVM）和朴素贝叶斯分类器（Naive Bayes）。

传统方法使用的文本表示方法大多是“词袋模型”。即只考虑文本中词的出现的频率，不考虑词的序列信息。

CNN（卷积神经网络），虽然出身于图像处理，但是它的思路，给我们提供了在NLP应用上的参考。

我们对于句子矩阵，是否可以用每一行表示一个单词，每个向量表示一个词，然后输入到卷积神经网络中进行处理呢？

本章任务就将介绍如何嵌入Word2vec词向量并使用CNN进行中文文本分类。



## 3 知识准备


### 3.1 什么是卷积？
简单地说就是一系列的输入信号进来之后，系统也会有一系列的输出。

但是并不是某一时刻的输出只对应该时刻的输入，而是根据系统自身的特征，每一个时刻的输出，都和之前的输入相关。

那么如果文本是一些列输入，我们当然希望考虑词和词的序列特征，比如“Tom 的 手机 ”，使用卷积，系统就会知道“手机是tom”的，而不是仅仅是一个“手机”。

或者更直观地理解，在CNN模型中，卷积就是拿**kernel**在图像上到处移动，每移动一次提取一次特征，组成feature map， 这个提取特征的过程，就是卷积。

![image](https://ask.qcloudimg.com/http-save/yehe-1269631/1ermo3rydu.gif)

如图所示是一个内核大小为3、步幅为1的二维卷积：
- 内核大小：内核大小定义了卷积的视野。二维的常见选择是3——即3x3像素。
- 步幅：步幅定义了遍历图像时内核的步长，默认值通常为1。
- padding：padding定义样本的边框如何处理。一（半）个padding卷积将保持空间输出尺寸等于输入尺寸，而如果内核大于1，则不加卷积将消除一些边界。
- 输入和输出通道：卷积层需要一定数量的输入通道（I），并计算出特定数量的输出通道（O）。可以通过I * O * K来计算这样一层所需的参数，其中K等于内核中的值的数量。


### 3.2 卷积神经网络如何应用在自然语言处理领域？

用卷积神经网络处理图像十分简单：单通道图像可以表示为一个矩阵，输入到CNN中，经过多组filter层和pooling层，得到图像的局部特征，然后进行相关任务。

那么对于一个句子甚至一篇文章来说呢？

这就联系到了我们之前学习的词向量的概念：我们用拼接词向量的方法，将一个句子表示成为一个矩阵，这里矩阵的每一行表示一个word，后面的步骤继续得到句子的特征向量，然后进行分类。

本章任务中将会用到的CNN结构如下图所示：

![image](https://github.com/cjymz886/text-cnn/raw/master/images/text_cnn.png)

下面将具体分析每一层结构的作用。

#### 3.2.1 输入层
可以把输入层理解成把一句话转化成了一个二维的矩阵：每一排是一个词的word2vec向量，纵向是这句话的每个词按序排列。

输入数据的size，也就是矩阵的size。

假设句子有 n 个词，vector的维数为 k ，那么这个矩阵就是 n×k 的。


#### 3.2.2 卷积层
输入层通过卷积操作得到若干个Feature Map，卷积窗口的大小为 h×k ，其中 h 表示纵向词语的个数，而 k 表示word vector的维数。通过这样一个大型的卷积窗口，将得到若干个列数为1的Feature Map。

#### 3.2.3 池化层
接下来的池化层，我们采了一种称为Max-over-time Pooling的方法。这种方法就是简单地从之前一维的Feature Map中提出最大的值，因为最大值被认为是按照这个kernel卷积后的最重要的特征。

可以看出，这种Pooling方式可以解决可变长度的句子输入问题（因为不管Feature Map中有多少个值，只需要提取其中的最大值）。

最终池化层的输出为各个Feature Map的最大值们，即一个一维的向量。

#### 3.2.4 全连接 + Softmax层
池化层的一维向量的输出通过全连接的方式，连接一个Softmax层，Softmax层可根据任务的需要设置（通常反映着最终类别上的概率分布）。

最终实现时，我们可以在倒数第二层的全连接部分上使用Dropout技术，即对全连接层上的权值参数给予L2正则化的限制。

这样做的好处是防止隐藏层单元自适应（或者对称），通过L2正则化对权值参数中各个元素的平方和然后再求平方根，可以防止模型过拟合。



## 4 任务实施

### 4.1 实施思路
上面讲了这么多理论细节，那么如何完成本节课的任务呢？

1. 准备数据集
2. 数据预处理
3. 利用Word2vec训练词向量
2. 配置CNN参数
3. 实现CNN
6. 训练模型
7. 测试模型
8. 预测模型

![image](https://github.com/gaussic/text-classification-cnn-rnn/raw/master/images/cnn_architecture.png)

### 4.2 实施步骤


#### 步骤1：数据集准备
本次任务我们使用[THUCNews](http://thuctc.thunlp.org/)，由清华大学自然语言处理实验室推出的中文文本分类数据集。

THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

本次训练只使用了其中的10个分类：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐。

每个分类6500条数据。

数据集划分如下：

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)


#### 步骤2：数据预处理
我们先读取文件数据，然后使用字符级的表示构建词汇表，接着读取词汇表转换成{词：id}表示，然后将分类目录固定，转换成{类别：id}表示。

下一步将数据集从文字转换为固定长度的id序列表示，最后通过batch_iter()为神经网络的训练准备经过shuffle的批次的数据。


```
import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
```

#### 步骤3：训练词向量模型

```
import logging
import time
import codecs
import sys
import re
import jieba
from gensim.models import word2vec


re_han= re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)") # 剪切文本标点符号

class Get_Sentences(object):
    '''
    Args:
         filenames: a list of train_filename,test_filename,val_filename
    Yield:
        word:a list of word cut by jieba
    '''

    def __init__(self,filenames):
        self.filenames= filenames

    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('\t')
                        assert len(line)==2
                        blocks=re_han.split(line[1])
                        word=[]
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend(jieba.lcut(blk))
                        yield word
                    except:
                        pass

def train_word2vec(filenames):
    '''
    use word2vec train word vector
    argv:
        filenames: a list of train_filename,test_filename,val_filename
    return:
        save word vector to config.vector_word_filename
    '''
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=1, workers=6)
    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

if __name__ == '__main__':
    train_filename=r'..\cnews.train.txt'  #train data
    test_filename=r'..\cnews.test.txt'    #test data
    val_filename=r'..\cnews.val.txt'      #validation data
    
    filenames=[train_filename,test_filename,val_filename]
    train_word2vec(filenames)
```

训练过程中控制台会不断输出当前的训练情况：

![image](https://s2.ax1x.com/2019/10/16/KFjCDA.png)

最后会输出训练用时：
![image](https://s2.ax1x.com/2019/10/16/KFjA4f.png)

本次训练词向量模型大概用了半小时。


#### 步骤4：配置CNN参数
我们来定义我们的CNN基本参数：

```
class TextConfig():

    embedding_size=100     # 词向量维度
    vocab_size=8000        # 词汇大小
    pre_trianing = None   # 使用由word2vec训练的字符向量

    seq_length=600         # 序列长度
    num_classes=10         # 类别数

    num_filters=128        # 卷积核数目
    filter_sizes=[2,3,4]   # 卷积核的大小


    keep_prob=0.5          # droppout保留比例
    lr= 1e-3               # 学习率
    lr_decay= 0.9          # 学习率衰减
    clip= 6.0              # 梯度限幅阈值
    l2_reg_lambda=0.01     # l2正则化lambda

    num_epochs=10          # 总迭代轮次
    batch_size=64         # 每批训练大小
    print_per_batch =100   # 输出结果
    
    train_filename=r'..\cnews.train.txt'  #train data
    test_filename=r'..\cnews.test.txt'    #test data
    val_filename=r'..\cnews.val.txt'      #validation data
    vocab_filename=r'..\vocab.txt'        #vocabulary
    vector_word_filename=r'..\vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz=r'..\vector_word.npz'   # save vector_word to numpy file
```

#### 步骤5：定义CNN

```
    def cnn(self):
        '''CNN模型'''
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn'):
            # CNN layer
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):

                    filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedding_inputs_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.outputs= tf.reshape(self.h_pool, [-1, num_filters_total])


        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1].value, self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b

            # 分类器
            self.prob=tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            # 损失函数，交叉熵
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            # 优化器
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            # 准确率
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
```

#### 步骤6：训练模型
下面我们开始训练我们的文本分类模型：

```
#encoding:utf-8
from __future__ import print_function
import text_model
import loader
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta

def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len



def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/textcnn'
    save_dir = './checkpoints/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 1000  # break training if not having improvement over 1000 iter
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, x_val, y_val)
                writer.add_summary(train_summaries, global_step)

                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch,improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TextConfig()
    filenames = [config.train_filename, config.test_filename, config.val_filename]
    if not os.path.exists(config.vocab_filename):
        build_vocab(filenames, config.vocab_filename, config.vocab_size)

    #read vocab and categories
    categories,cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    model = TextCNN(config)
    train()
```

本实验经过5轮的迭代，满足终止条件结束，可以看到在global_step=2400时在验证集得到最佳效果97.2%:
![image](https://s2.ax1x.com/2019/10/16/KFvFsJ.md.png)
![image](https://s2.ax1x.com/2019/10/16/KFvkL9.png)
![image](https://s2.ax1x.com/2019/10/16/KFvCzF.png)
![image](https://s2.ax1x.com/2019/10/16/KFjOqs.png)
![image](https://s2.ax1x.com/2019/10/16/KFjLrj.png)

#### 步骤7：测试模型

```
from __future__ import print_function
import text_model
import loader
import text_predict
import  tensorflow as tf
from sklearn import metrics
import sys
import os
import time
import numpy as np
from datetime import timedelta


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = loader.batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def test():
    print("Loading test data...")
    t1=time.time()
    x_test,y_test=loader.process_file(config.test_filename,word_to_id,cat_to_id,config.seq_length)

    session=tf.Session()
    session.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess=session,save_path=save_path)

    print('Testing...')
    test_loss,test_accuracy = evaluate(session,x_test,y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(test_loss, test_accuracy))

    batch_size=config.batch_size
    data_len=len(x_test)
    num_batch=int((data_len-1)/batch_size)+1
    y_test_cls=np.argmax(y_test,1)
    y_pred_cls=np.zeros(shape=len(x_test),dtype=np.int32)

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        feed_dict={
            model.input_x:x_test[start_id:end_id],
            model.keep_prob:1.0,
        }
        y_pred_cls[start_id:end_id]=session.run(model.y_pred_cls,feed_dict=feed_dict)

    #evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    print("Time usage:%.3f seconds...\n"%(time.time() - t1))

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = text_model.TextConfig()
    filenames = [config.train_filename, config.test_filename, config.val_filename]
    if not os.path.exists(config.vocab_filename):
        loader.build_vocab(filenames, config.vocab_filename, config.vocab_size)
    #read vocab and categories
    categories,cat_to_id = loader.read_category()
    words,word_to_id = loader.read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        loader.export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = loader.get_training_word2vec_vectors(config.vector_word_npz)
    model = text_model.TextCNN(config)

    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')
    test()
```

在测试集上运行结果如下：

![image](https://s2.ax1x.com/2019/10/16/KFvgYV.png)

对测试数据集显示，test_loss=0.1，test_accuracy=97.23%，其中“体育”类测试为100%，整体的precision=recall=F1=97%。

#### 步骤8：预测模型

```
#encoding:utf-8
from ch7.text_model import *
import  tensorflow as tf
import tensorflow.contrib.keras as kr
import os
import numpy as np
import jieba
import re
import heapq
import codecs



def predict(sentences):
    config = TextConfig()
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)
    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')

    _,word_to_id=read_vocab(config.vocab_filename)
    input_x= process_file(sentences,word_to_id,max_length=config.seq_length)
    labels = {0:'体育',
              1:'财经',
              2:'房产',
              3:'家居',
              4:'教育',
              5:'科技',
              6:'时尚',
              7:'时政',
              8:'游戏',
              9:'娱乐'
              }

    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
    }
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob=session.run(model.prob, feed_dict=feed_dict)
    y_prob=y_prob.tolist()
    cat=[]
    for prob in y_prob:
        top2= list(map(prob.index, heapq.nlargest(1, prob)))
        cat.append(labels[top2[0]])
    tf.reset_default_graph()
    return  cat

def sentence_cut(sentences):
    """
    Args:
        sentence: a list of text need to segment
    Returns:
        seglist:  a list of sentence cut by jieba
    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    seglist=[]
    for sentence in sentences:
        words=[]
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                words.extend(jieba.lcut(blk))
        seglist.append(words)
    return  seglist


def process_file(sentences,word_to_id,max_length=600):
    """
    Args:
        sentence: a text need to predict
        word_to_id:get from def read_vocab()
        max_length:allow max length of sentence
    Returns:
        x_pad: sequence data from  preprocessing sentence
    """
    data_id=[]
    seglist=sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    return x_pad


def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to idbuild_vocab
    """
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]

if __name__ == '__main__':
    print('predict random five samples in test data.... ')
    import random
    sentences=[]
    labels=[]
    with codecs.open(r'..\cnews.test.txt','r',encoding='utf-8') as f:
        sample=random.sample(f.readlines(),5)
        for line in sample:
            try:
                line=line.rstrip().split('\t')
                assert len(line)==2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass
    cat=predict(sentences)
    for i,sentence in enumerate(sentences,0):
        print ('----------------------the text-------------------------')
        print (sentence[:50]+'....')
        print('the orginal label:%s'%labels[i])
        print('the predict label:%s'%cat[i])
```

运行:python text_predict.py

随机从测试数据中挑选了五个样本，输出原文本和它的原文本标签和预测的标签，下图中5个样本预测的都是对的：

![image](https://s2.ax1x.com/2019/10/16/KFvTT1.png)




## 5 任务拓展
CNN模型根据词向量的不同分为四种：

1. CNN-rand，所有的词向量都随机初始化，并且作为模型参数进行训练。
1. CNN-static，即用word2vec预训练好的向量（Google News），在训练过程中不更新词向量，句中若有单词不在预训练好的词典中，则用随机数来代替。
1. CNN-non-static，根据不同的分类任务，进行相应的词向量预训练。
1. CNN-multichannel，两套词向量构造出的句子矩阵作为两个通道，在误差反向传播时，只更新一组词向量，保持另外一组不变。


其中，对于未登录词的vector，可以用0或者随机小的正数来填充。
## 6 任务实训


### 6.1 实训目的
1. 掌握卷积神经网络的结构和原理
2. 学会使用卷积神经网络训练文本（基于TensorFlow框架）

### 6.2 实训内容
动手修改CNN配置中的参数，重新训练属于你自己的模型，对比一下不同参数设置会对训练出来的模型的准确度产生什么影响吧！

