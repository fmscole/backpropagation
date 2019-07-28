import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
BATCH_SIZE = 500

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out=torch.nn.Linear(in_features=64,out_features=10)

    def forward(self,x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output,(h_n,c_n)=self.rnn(x)
        # print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep=h_n[-1,:,:]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x=self.out(output_in_last_timestep)
        return x

if __name__ == "__main__":
    # 1. 加载数据
    training_dataset = torchvision.datasets.MNIST("./mnist_dataset", train=True,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = Data.DataLoader(dataset=training_dataset,
                                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # showSample(dataloader)
    test_data=torchvision.datasets.MNIST(root="./mnist_dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=False)
    test_dataloader=Data.DataLoader(
        dataset=test_data,batch_size=1000,shuffle=False,num_workers=2)
    testdata_iter=iter(test_dataloader)
    test_x,test_y=testdata_iter.next()
    test_x=test_x.view(-1,28,28)
    # 2. 网络搭建
    net=RNN()
    # 3. 训练
    # 3. 网络的训练（和之前CNN训练的代码基本一样）
    optimizer=torch.optim.Adam(net.parameters(),lr=0.001)
    loss_F=torch.nn.CrossEntropyLoss()
    for epoch in range(10): # 数据集只迭代一次
        for step, input_data in enumerate(dataloader):
            x,y=input_data
            pred=net(x.view(-1,28,28))
    
            loss=loss_F(pred,y) # 计算loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50==49: # 每50步，计算精度
                with torch.no_grad():
                    test_pred=net(test_x)
                    prob=torch.nn.functional.softmax(test_pred,dim=1)
                    pred_cls=torch.argmax(prob,dim=1)
                    acc=(pred_cls==test_y).sum().numpy()/pred_cls.size()[0]
                    print(f"{epoch}-{step}: accuracy:{acc}")