class Net(object):
    def __init__(self, seq):
        self.seq=seq
        self.bkseq=seq[::-1]

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
