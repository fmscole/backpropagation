class Net(object):
    def __init__(self, seq,input_shape):
        self.seq=seq
        self.bkseq=seq[::-1]
        #最后一层一般与损失函数结合一起计算，不需要反向传播
        self.bkseq=self.bkseq[1:]
        self.OutShape(input_shape)
    def OutShape(self,input_shape):
        out_shape=input_shape
        for layer in self.seq:
            out_shape=layer.OutShape(out_shape)
        return out_shape

    def forward(self,imgs,training=True):
        out=imgs
        for layer in self.seq:
            out=layer.forward(out)
        return out

    def backward(self,grad,training=True):
        for layer in self.bkseq:
            grad=layer.backward(grad)
        return grad

    def Gradient(self,alpha=0.001, weight_decay=0.001):
        for layer in self.seq:
            try:
                layer.gradient(alpha=alpha, weight_decay=weight_decay)
            except :
                pass 
