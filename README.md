# backward

#### 项目介绍
反向传播学习代码。
作为知乎专栏：反向传播 https://zhuanlan.zhihu.com/fmscole 的配套代码。
源码都是从各个地方收集，再有一些或多或少的修改。
特点就是都没使用神经网络库（比如tensorflow，pytorch等），主要使用numpy,numba。   
目的不是重新造轮子，而是为了学习一些基本算法的细节，为了算法完全的透明，复现了算法效果才算验证了真的理解了算法。   
所以，性能、稳定性、扩展性都不是目标，为了聚焦对算法本身的理解，所有代码都尽可能的简单易读，这是最大原则。
但是python的性能有的时候令人抓狂，所以也会在一些性能瓶颈的的地方使用numba加速，同时也会保留易读版本。

1.FCN、FCNet都是全连接网络。
  FCN是最简单朴素的代码。来自
  https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork \
  FCNet改写成分层结构，并测试了BatchNormal 层。

2.CNN是卷积神经网络。
  来自   
  https://github.com/wuziheng/CNN-Numpy    
  https://github.com/leeroee/MNN    
  https://github.com/sebgao/cTensor
  
3.RNN是循环神经网络。
  来自https://github.com/qixianbiao/RNN
  
4.LSTM长短记忆神经网络。
  来自https://github.com/nicodjimenez/lstm


#### 运行说明

1. 在目录下“backprapagation”下运行。
   如果是用pycharm或者vscode，在打开文件夹时，选“backprapagation”文件夹。
   如果用jupyter notebook，则在backprapagation路径下启动jupyter notebook。
   如果进入子目录运行程序，路径相关的代码需要您自己修改，否则不能正确运行。
   如果您对路径的设置熟悉，这些修改也是简单的。
