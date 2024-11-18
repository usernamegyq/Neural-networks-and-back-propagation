# Neural-networks-and-back-propagation
## 任务1<br>
  在numpy_BP_1.py中进行了任务1的实现，包括采用numpy实现BP神经网络、在half moon上进行分类以及可视化<br>
## 任务2<br>
  在numpy_BP_2.py中实现了基于numpy实现BP神经网络的mnist图像分类<br>
  进行的参数调试与选择包括随机初始化权重、He初始化权重、学习率的调节、正则化的调节、激活函数的调整、分批次训练、隐藏层的参数调整<br>
  最后在隐藏层层数为1、节点数为128、学习率为0.01、迭代次数为10次、batch_size为60、He初始化权重的情况下在测试集上的准确率可达97%左右<br>
## 任务3<br>
  在pytorch_1.py中实现了lenet、vgg、resnet的模型并进行了mnist图像分类，在每个模型的实现过程中都减少了模型的复杂度以缩短训练时间。最终三个模型都在0.001的学习率、迭代次数为10次的情况下在测试集上达到97%以上的准确率，其中vgg模型的准确率相对偏高，可达98%，但训练时间与lenet和renet比相对较长。<br>
  在transformers_1.py中构建了一个极小的vision transformer网络，参数如下：<br>
    image_size = 28
    patch_size = 14
    in_channels = 1
    embedding_dim = 64
    feedforward_dim = 128
    num_heads = 4
    lr = 1e-3
    epchos = 10
  利用交叉熵损失函数和Adam优化器进行训练，经过训练后在mnist测试集上的准确度可达97.45%<br>
  模型的参数还可以进一步调整以提高学习率，包括采用不同的优化器，提高迭代次数，以及学习率的调整等<br>
## 任务4<br>
  在该任务中研究了voc数据集的图片和对应标注，但从百度网盘下载的数据集中Annotations只有图像提取的结果，缺少对应的标注，且在train_seg.py中的类别只有person、bird、car、cat、plane五种，没法进行20类的图像分类，因此我从网上重新下载了一个voc的数据集‘VOC’，并在该数据集中选取6000张作为训练集，624张作为测试集，以对应题目要求（原本我采用了完整的voc数据集进行训练与测试，但训练时间过长，且训练效果不佳，因此减少数据集大小，以缩短训练时间，提高训练效果）<br>
  voc_training_2.py中采用vit模型进行图像分类实验,vit模型在任务3的基础上增加了模型的复杂度和层数，同时增大参数以提高模型性能<br>
  voc_training_3.py中采用resnet-18和lenet的预训练模型进行图像分类实验<br>
  voc_training_4.py中采用vgg模型，在任务3的vgg模型的基础上增加卷积层和全连接层层数，并导入vgg16的预训练权重，冻结了卷积层，仅训练全连接层，以缩短训练时间<br>
