[TOC]
# 任务n：学习理解人体姿态估计相关知识
## 1.任务目标
<!-- 1. 
2. 
3. 
4.  -->
- 学习理解人体姿态估计的概念和理论
- 了解人体姿态估计领域常用数据集
- 掌握人体姿态估计领域的衡量标准以及损失函数
- 学会使用相关的人体姿态估计算法

## 2.任务描述
- 从自然实况入手，一步步从计算机视觉的角度去理解如何做人体姿态的估计
- 以二维的人体姿态估计为切入点,讲解围绕二维人体姿态估计的相关衡量标准以及损失函数
- 动手实践一个人体姿态估计算法,观察预测结果


## 3.知识准备

### 相关数据集
#### Benchmark
<!-- 还可以给每个数据集加点简单的描述 -->
- 单人估计: [MPII](http://human-pose.mpi-inf.mpg.de/), [FLIC](https://bensapp.github.io/flic-dataset.html), [LSP](http://sam.johnson.io/research/lsp.html), [LIP](http://sysu-hcp.net/lip/)

- 多人关键点预测: [COCO](http://cocodataset.org/#keypoints-2019), [CrowdPose](http://cocodataset.org/#keypoints-2019)

- 视频：[PoseTrack](https://posetrack.net/)

- 三维人体：[Human3.6M](http://vision.imar.ro/human3.6m/description.php), [DensePose](http://densepose.org/)


### 评估指标
<!-- 此处考虑贴入公式或者图片 -->
- 基于对象关键点相似度（OKS）的mAP：
- $$\frac{\sum}{2}$$
- AP(average precision)
- AP Across Scales
- AR(average recall)
- AR Across Scales

### 损失函数



### 自下而上和自上而下的检测思想
<!-- 加上小段的文字介绍 -->
#### 自下而上
- Mask R-CNN, CPN, MSPN
- 高性能（良好的本地化能力），高召回率


#### 自上而下
- Openpose, Associative Embeding
- 简洁的框架，可能更快

### 挑战
- 模棱两可的外观
- 拥挤人群，遮挡现象
- 实时估计的速度

### 相关算法



## 4. 任务实施
### 4.1 实施思路


### 4.2 实施步骤


## 5.任务拓展



## 6. 任务实训
### 6.1 实训目的


### 6.2 实训内容

### 6.3 示例代码



```

```