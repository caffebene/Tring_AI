# 任务2：TensorFlow基本用法
## 1 任务目标
- 理解并掌握TensorFlow的基础概念
- 熟悉TensorFlow的工作流程
- 利用TensorFlow构建简单的计算

## 2 任务描述
      
在本任务中，我们将从概念出发，初步认识TensorFlow框架的基本用法以及工作流程，再辅以一些例子帮助理解，最终达到能够初步运用TensorFlow去构建一些简单计算的目的。

## 3 知识准备
- 使用图 (graph) 来表示计算任务
- 在被称之为会话 (Session) 的上下文 (context) 中执行图
- 使用 tensor 表示数据
- 通过变量 (Variable) 来记录状态.
- 使用 feed 和 fetch 可以为任意的计算赋值或者从其中获取数据

### 3.1 综述
- TensorFlow 是一个使用 python 的编程框架，使用图来表示计算任务。
- 图中的节点被称之为 op (operation 的缩写)。 
- 一个 op 获得 0 个或多个 Tensor，执行计算，产生 0 个或多个 Tensor。
- 每个 Tensor 是一个类型化的多维数组。例如， 你可以将一小组图像集表示为一个四维浮点数数组， 这四个维度分别是 [batch, height, width, channels]。

### 3.2 计算图
TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段。在构建阶段，op 的执行步骤 被描述成一个图。在执行阶段, 使用会话执行执行图中的 op。

例如，通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op。

### 3.3 构建图
构建图的第一步，是创建源 op (source op)。源 op 不需要任何输入，例如常量 (Constant)。 源 op 的输出被传递给其它 op 做运算。

TensorFlow 库中, op 构造器的返回值代表被构造出的 op 的输出，这些返回值可以传递给其它 op 构造器作为输入。

TensorFlow 库有一个默认图 (default graph)，op 构造器可以为其增加节点。这个默认图对许多程序来说已经足够用了。当然也可以自定义图。


```
import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)

```
对于上述程序，默认图现在有三个节点，两个 constant() op, 和一个matmul() op。为了真正进行矩阵相乘运算, 并得到矩阵乘法的结果, 你必须在会话里启动这个图。

### 3.4 在一个会话中启动图

构造阶段完成后，才能启动图。启动图的第一步是创建一个 Session 对象，如果无任何创建参数，会话构造器将启动默认图。


```
# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
```

Session 对象在使用完后需要关闭以释放资源。除了显式调用 close 外, 也可以使用 "with" 代码块来自动完成关闭动作。


```
with tf.Session() as sess:
  result = sess.run([product])
  print result
```

### 3.4 交互式使用
文档中的 Python 示例使用一个会话 Session 来启动图，并调用 Session.run() 方法执行操作。

为了便于使用诸如 IPython 之类的 Python 交互环境，可以使用 InteractiveSession 代替 Session 类，使用 Tensor.eval() 和 Operation.run() 方法代替 Session.run()。 这样可以避免使用一个变量来持有会话。


```
# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
print sub.eval()
# ==> [-2. -1.]
```

### 3.5 Tensor
TensorFlow 程序使用 tensor 数据结构来代表所有的数据，计算图中， 操作间传递的数据都是 tensor。你可以把 tensor 看作是一个 n 维的数组或列表。一个 tensor 包含一个静态类型 rank 和一个 shape。 

### 3.6 变量
变量记录图执行过程中的状态信息。下面的例子演示了如何使用变量实现一个简单的计数器。


```
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)
  # 打印 'state' 的初始值
  print sess.run(state)
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print sess.run(state)

# 输出:

# 0
# 1
# 2
# 3

```

代码中 assign() 操作是图所描绘的表达式的一部分，正如 add() 操作一样。所以在调用 run() 执行表达式之前，它并不会真正执行赋值操作。

通常会将一个统计模型中的参数表示为一组变量。例如, 你可以将一个神经网络的权重作为某个变量存储在一个 tensor 中。在训练过程中, 通过重复运行训练图，更新这个 tensor。

### 3.7 Fetch
为了取回操作的输出内容，可以在使用 Session 对象的 run() 调用执行图时，传入一些 tensor，这些 tensor 会帮助你取回结果。在之前的例子里，我们只取回了单个节点 state, 但是你也可以取回多个 tensor:

```
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

#构造求和操作
intermed = tf.add(input2, input3)
#构造乘法操作
mul = tf.mul(input1, intermed)

with tf.Session():
  result = sess.run([mul, intermed])
  print result

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

```
需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。

###  3.8 Feed

上述示例在计算图中引入了 tensor，以常量或变量的形式存储。TensorFlow 还提供了 feed 机制，该机制可以临时替代图中的任意操作中的 tensor 的值，经常用作输入时的“赋值”。

feed 使用一个 tensor 值临时替换一个操作的输出结果。你可以提供 feed 数据作为 run() 调用的参数。 feed 只在调用它的方法内有效，方法结束，feed 就会消失。 最常见的用例是将某些特殊的操作指定为 "feed" 操作， 标记的方法是使用 tf.placeholder() 为这些操作创建占位符。占位符顾名思义就是先设置好数据的位置，等待数据的输入。


```
input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

# 输出:
# [array([ 14.], dtype=float32)]
```

## 4 任务实施
### 4.1 实施内容
- 利用TensorFlow的计算图来实现矩阵的组合四则运算
- 构造一个2X3的矩阵和3X2的矩阵，将它们相乘的结果再与一个2X2的矩阵求和
- 要求先将相乘的结果赋值给一个变量，再用变量进行求和

### 4.2 实施步骤
- 矩阵的构造可以利用python的嵌套list
- 例如2X3的矩阵：a = tf.constant([[2,3,1],[3,1,2]])
- 矩阵乘法可以直接调用函数，将矩阵当成参数传入：例如矩阵a X b：product = tf.matmul( a, b )
- 赋值函数 tf.assign(a,b) ，表示将b的值赋给a
- 初始化变量x可以用：init = tf.initialize_all_variables()，然后直接在会话中run（init）即可

### 实施代码

```
import tensorflow as tf

#构造矩阵
a = tf.constant([[2,3,1],[3,1,2]])
b = tf.constant([[1,3],[3,4],[1,2]])
c = tf.constant([[1,2],[2,1]])

#构造2X2的变量
d = tf.Variable([[0,0],[0,0]])

#矩阵乘法
product = tf.matmul( a, b )

#赋值
update = tf.assign( d, product )

#加法运算
result = tf.add( d, c )

#初始化变量
init = tf.initialize_all_variables()

#在会话中运行
with tf.Session() as sess:
    sess.run(init)
    sess.run(update)
    print(sess.run(d))
    print(sess.run(result))
```

参考运行结果

```
[[12 20]
 [ 8 17]]
 [[13 22]
 [10 18]]
```


## 5 任务拓展
 在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU). 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作。
 
 如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的。 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行。 with tf.Device() 语句用来指派特定的 CPU 或 GPU 执行操作:


```
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

设备用字符串进行标识. 目前支持的设备包括:
- "/cpu:0": 机器的 CPU.
- "/gpu:0": 机器的第一个 GPU, 如果有的话.
- "/gpu:1": 机器的第二个 GPU, 以此类推.

## 6 任务实训
示例代码展示了矩阵的乘法和加法，接下来大家可以仿照示例去实现剩余的矩阵运算，比如矩阵的除法、减法、点乘。

任务提示：
- 除法函数：tf.divide(a,b)
- 减法函数：tf.subtract(a,b)
- 点乘：tf.multiply(a,b)
