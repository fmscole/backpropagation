#python3
import struct
from glob import glob
import os
import numpy as np

class my_data_set:
    def next_batch(self,batch_size):
        i=self.i
        if i + batch_size<=self.size:
            batch_xs1 = self.images[:,i :i +  batch_size]
            batch_ys1 = self.labels[:,i :i + batch_size]
            self.i=self.i+batch_size
            if self.i==self.size:
                self.i=0
                # print (self.i,batch_xs1.shape[0])
            return batch_xs1,batch_ys1
        if i <self.size:
            batch_xs1 = self.images[:,i:]
            batch_ys1 = self.labels[:,i :]
            self.i =batch_size-self.size+self.i
            batch_xs1 = np.concatenate([batch_xs1,self.images[:,0:self.i ]])
            batch_ys1 =np.concatenate([batch_ys1,self.labels[:,0:self.i ]])
            # print (self.i,batch_xs1.shape[0])
            return batch_xs1,batch_ys1

    def load_mnist(self, kind='train'):
        if kind=='train':
            images_path =r'./mnist_dataset/train-images.idx3-ubyte'
            labels_path =r'./mnist_dataset/train-labels.idx1-ubyte' 
        else: 
            images_path =r'./mnist_dataset/t10k-images.idx3-ubyte' 
            labels_path =r'./mnist_dataset/t10k-labels.idx1-ubyte' 
        print(images_path, labels_path)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))

            labels = np.fromfile(lbpath, dtype=np.uint8)
            x = np.zeros((labels.shape[0], 10))
            for i in range(labels.shape[0]):
                x[i][labels[i]] = 1
            labels = np.array(x)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                imgpath.read(16))
            images = np.fromfile(imgpath,
                                dtype=np.uint8).reshape(len(labels), 784)
            images=np.array(images)/255
            # print(images.shape)
        im = np.array(images, ndmin=2).T
        lb = np.array(labels, ndmin=2).T
        self.images=im
        self.labels=lb
        self.size=im.shape[1]
        self.i=0
        # return images, labels

if __name__=="__main__":
    mydata=my_data_set()
    mydata.load_mnist( kind='train')
    images,labels=mydata.next_batch(1000)
    print(images.shape)
    print(labels.shape)
    