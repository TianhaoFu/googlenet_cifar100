# googlenet_cifar100

googlenet的Pytorch实现。

## 1-数据集：cifar-100

基本情况与CIFAR-10一样。但CIFAR-100的类别更精细：CIFAR-100中图像被分为100个类，100个类又被分为20个超类

## 2-超参


Model structure：	Googlenet

epoch(lr = 0.01)：	60

epoch(lr = 0.002)：60

epoch(lr = 0.0004)：	40

epoch(lr = 0.00008)：	40

total epoch：	200（41） 【实际训练时因google colabGPU只能连续90min训练模型，因此模型训练到accuracy为70%左右(41个epoch)即暂停训练】

Batchsize：	32

Optimizer：	Momentum

Nesterov momentum： 	0.9

Learning rate(initial)：	0.01

Warm up epoch：	1

Learning rate Multistep decay(gamma)：	0.2

Weight decay(Regularization)：	5e-4

time：	8369.40672s
