[TOC]
# 任务7：AlexNet与cifar10数据集

## 1 任务目标

1. 认识学习经典网络 AlexNet 与 数据集 cifar-10

2. 仿照 AlexNet 实现对 cifar-10 数据集的识别



## 2 任务描述

经过前两个任务的学习，相信大家对于 CNN 已经有了一定的了解。在本次任务中，将介绍一种经典的卷积神经网络 AlexNet 以及 cifar-10 数据集，并仿照 AlexNet 实现对 cifar-10 数据集的识别。


## 3 知识准备

### 3.1 AlexNet

图像识别一直是机器学习中非常重要的一个研究内容, 2010年开始举办的[ILSVRC](http://image-net.org/challenges/LSVRC/)比赛更是吸引了无数的团队. 这个比赛基于一个百万量级的图片数据集, 提出一个图像1000分类的挑战. 前两年在比赛中脱颖而出的都是经过人工挑选特征, 再通过`SVM`或者`随机森林`这样在过去十几年中非常成熟的机器学习方法进行分类的算法. 

在2012年, 由 [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/), [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/), [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/)提出了一种使用卷积神经网络的方法, 以 [0.85](http://image-net.org/challenges/LSVRC/2012/results.html#abstract) 的`top-5`正确率一举获得当年分类比赛的冠军, 超越使用传统方法的第二名10个百分点, 震惊了当时的学术界, 从此开启了人工智能领域的新篇章.

这次的课程我们就来复现一次`AlexNet`, 首先来看它的网络结构

![](https://ae01.alicdn.com/kf/Hb0a300b1c2884918802fdf6a2f151e0fF.png)

从这个图我们可以很清楚地看到Alexnet的整个网络结构是由5个卷积层和3个全连接层组成的，深度总共8层。 下面就让我们来尝试仿照这个结构来解决[cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)分类问题.


### 3.2 CIFAR-10

该数据集共有60000张彩色RGB图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

下面这幅图就是列举了10各类，每一类展示了随机的10张图片：


![](https://ae01.alicdn.com/kf/H3a4f489a026148af99584cd0929fdc4as.png)


## 4 任务实施

### 4.1 实施思路

1.获取数据集（已下载至平台中），用语句读取即可

2.定义权重和偏置，封装卷积、池化、全连接的构造

3.定义损失函数与正确率

4.构造训练（已封装好训练过程，同学们调用即可）

### 4.2 实施步骤


```
import tensorflow as tf
from utils import cifar10_input
```


```
# 我们定义一个批次有64个样本
batch_size = 64

# 获取训练集
# 在使用随机梯度下降法的时候, 训练集要求打乱样本
train_imgs, train_labels = cifar10_input.inputs(eval_data=False,data_dir='cifar10_data/cifar-10-batches-bin/',batch_size=batch_size, shuffle=True)

# 获取测试集
# 测试集不需要打乱样本
val_imgs, val_labels = cifar10_input.inputs(eval_data=True,data_dir='cifar10_data/cifar-10-batches-bin/', batch_size=batch_size, shuffle=False)

train_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN    # 训练样本的个数
val_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL       # 测试样本的个数
```



```
# 像之前一样, 我们构造几个生成变量的函数
def variable_weight(shape, stddev=5e-2):
    init = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(shape=shape, initializer=init, name='weight')

def variable_bias(shape):
    init = tf.constant_initializer(0.1)
    return tf.get_variable(shape=shape, initializer=init, name='bias')
```

前面的课程大家已经见过如何使用卷积和池化了, 但是由于tensorflow提供的接口是底层的, 为了方便, 我们写一个上层的接口来调用。



```
def conv(x, ksize, out_depth, strides, padding='SAME', act=tf.nn.relu, scope='conv_layer', reuse=None):
    """构造一个卷积层
    Args:
        x: 输入
        ksize: 卷积核的大小, 一个长度为2的`list`, 例如[3, 3]
        output_depth: 卷积核的个数
        strides: 卷积核移动的步长, 一个长度为2的`list`, 例如[2, 2]
        padding: 卷积核的补0策略
        act: 完成卷积后的激活函数, 默认是`tf.nn.relu`
        scope: 这一层的名称(可选)
        reuse: 是否复用
    
    Return:
        out: 卷积层的结果
    """
    # 这里默认数据是NHWC输入的
    in_depth = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(scope, reuse=reuse):
        # 先构造卷积核
        shape = ksize + [in_depth, out_depth]
        with tf.variable_scope('kernel'):
            kernel = variable_weight(shape)
            
        strides = [1, strides[0], strides[1], 1]
        # 生成卷积
        conv = tf.nn.conv2d(x, kernel, strides, padding, name='conv')
        
        # 构造偏置
        with tf.variable_scope('bias'):
            bias = variable_bias([out_depth])
            
        # 和偏置相加
        preact = tf.nn.bias_add(conv, bias)
        
        # 添加激活层
        out = act(preact)
        
        return out
```


```
def max_pool(x, ksize, strides, padding='SAME', name='pool_layer'):
    """构造一个最大值池化层
    Args:
        x: 输入
        ksize: pooling核的大小, 一个长度为2的`list`, 例如[3, 3]
        strides: pooling核移动的步长, 一个长度为2的`list`, 例如[2, 2]
        padding: pooling的补0策略
        name: 这一层的名称(可选)
    
    Return:
        pooling层的结果
    """
    return tf.nn.max_pool(x, [1, ksize[0], ksize[1], 1], [1, strides[0], strides[1], 1], padding, name=name)
```


```
def fc(x, out_depth, act=tf.nn.relu, scope='fully_connect', reuse=None):
    """构造一个全连接层
    Args:
        x: 输入
        out_depth: 输出向量的维数
        act: 激活函数, 默认是`tf.nn.relu`
        scope: 名称域, 默认是`fully_connect`
        reuse: 是否需要重用
    """
    in_depth = x.get_shape().as_list()[-1]
    
    # 构造全连接层的参数
    with tf.variable_scope(scope, reuse=reuse):
        # 构造权重
        with tf.variable_scope('weight'):
            weight = variable_weight([in_depth, out_depth])
            
        # 构造偏置项
        with tf.variable_scope('bias'):
            bias = variable_bias([out_depth])
        
        # 一个线性函数
        fc = tf.nn.bias_add(tf.matmul(x, weight), bias, name='fc')
        
        # 激活函数作用
        out = act(fc)
        
        return out
```


有了上面这些上层函数之后, 我们就可以轻松构建我们所需的网络了。


```
def alexnet(inputs, reuse=None):
    """构建 Alexnet 的前向传播
    Args:
        inpus: 输入
        reuse: 是否需要重用
        
    Return:
        net: alexnet的结果
    """
    # 首先我们声明一个变量域`AlexNet`
    with tf.variable_scope('AlexNet', reuse=reuse):
        # 第一层是 5x5 的卷积, 卷积核的个数是64, 步长是 1x1, padding是`VALID`
        net = conv(inputs, [5, 5], 64, [1, 1], padding='VALID', scope='conv1')
        
        # 第二层是 3x3 的池化, 步长是 2x2, padding是`VALID`
        net = max_pool(net, [3, 3], [2, 2], padding='VALID', name='pool1')
        
        # 第三层是 5x5 的卷积, 卷积核的个数是64, 步长是 1x1, padding是`VALID`
        net = conv(net, [5, 5], 64, [1, 1], scope='conv2')
        
        # 第四层是 3x3 的池化, 步长是 2x2, padding是`VALID`
        net = max_pool(net, [3, 3], [2, 2], padding='VALID', name='pool2')
        
        # 将矩阵拉长成向量
        net = tf.reshape(net, [-1, 6*6*64])
        
        # 第五层是全连接层, 输出个数为384
        net = fc(net, 384, scope='fc3')
        
        # 第六层是全连接层, 输出个数为192
        net = fc(net, 192, scope='fc4')
        
        # 第七层是全连接层, 输出个数为10, 注意这里不要使用激活函数
        net = fc(net, 10, scope='fc5', act=tf.identity)
        
        return net
```
通过 alexnet 构建训练和测试的输出


```
train_out = alexnet(train_imgs)
```

注意当再次调用 alexnet 函数时, 如果要使用之前调用时产生的变量值, 必须要重用变量域


```
val_out = alexnet(val_imgs, reuse=True)
```

- 定义损失函数

这里真实的 labels 不是一个 one_hot 型的向量, 而是一个数值, 因此我们使用 


```
with tf.variable_scope('loss'):
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_out, scope='train')
    val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels, logits=val_out, scope='val')
```

- 定义正确率
```
with tf.name_scope('accuracy'):
    with tf.name_scope('train'):
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), train_labels), tf.float32))
    with tf.name_scope('train'):
        val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_out, axis=-1, output_type=tf.int32), val_labels), tf.float32))
```
- 构造训练

```
#学习率0.01
lr = 0.01

opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
train_op = opt.minimize(train_loss)
```
在这里我们已经将训练过程封装好了, 感兴趣的同学可以进入 train.py 查看

```,
#这里为了控制时间，预设的迭代次数为4000，如有条件可设为20000次，准备率会有显著提升。
from utils.learning import train
train(train_op, train_loss, train_acc, val_loss, val_acc, 4000, batch_size)
```


## 5 任务拓展

第一个典型的CNN是LeNet5网络结构，但是第一个引起大家注意的网络却是AlexNet，也就是文章《ImageNet Classification with Deep Convolutional Neural Networks》介绍的网络结构。

这篇文章的网络是在2012年的ImageNet竞赛中取得冠军的一个模型整理后发表的文章。作者是多伦多大学的Alex Krizhevsky等人。Alex Krizhevsky其实是Hinton的学生，这个团队领导者是Hinton，那么Hinton是谁呢？这就要好好说说了，网上流行说 Hinton， LeCun和Bengio是神经网络领域三巨头，LeCun就是LeNet5作者(Yann LeCun)，昨天的文章就提到了这个人。

而今天的主角虽然不是Hinton，但却和他有关系，这篇的论文第一作者是Alex，所以网络结构称为AlexNet。这篇论文很有意思，因为我读完这篇论文之后，没有遇到比较难以理解的地方，遇到的都是之前学过的概念，比如Relu，dropout。之前学的时候只知道Relu是怎么一回事，今天才知道它真正的来源。这篇文章在2012年发表，文章中的模型参加的竞赛是ImageNet LSVRC-2010，该ImageNet数据集有1.2 million幅高分辨率图像，总共有1000个类别。测试集分为top-1和top-5，并且分别拿到了37.5%和17%的error rates。

这样的结果在当时已经超过了之前的工艺水平。AlexNet网络结构在整体上类似于LeNet，都是先卷积然后在全连接。但在细节上有很大不同。AlexNet更为复杂。AlexNet有60 million个参数和65000个神经元，五层卷积，三层全连接网络，最终的输出层是1000通道的softmax。

AlexNet利用了两块GPU进行计算，大大提高了运算效率，并且在ILSVRC-2012竞赛中获得了top-5测试的15.3%error rate， 获得第二名的方法error rate 是 26.2%，可以说差距是非常的大了，足以说明这个网络在当时给学术界和工业界带来的冲击之大。


## 6 任务实训


### 6.1 实训目的

加深对于卷积神经网络的理解和实践

### 6.2 实训内容

可仿照AlexNet设置更加复杂和更有深度的 CNN ，实现对 cifar-10 数据集的识别。

