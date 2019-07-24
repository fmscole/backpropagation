import numpy as np
from RNN.rnn import RNN,sigmoid

#构建nums组训练数据，每组9个数据（9列行）
def getrandomdata(nums):
    x = np.zeros([nums, 10, 9], dtype=float)
    y = np.zeros([nums, 10, 9], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        for j in range(9):
            if tmpi < 8:
                x[i, tmpi, j], y[i, tmpi+1, j] = 1.0, 1.0
                tmpi = tmpi+1
            else:
                x[i, tmpi, j], y[i, 0, j] = 1.0, 1.0
                tmpi = 0
    return x, y



def test(nums):
    testx = np.zeros([nums, 10], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        testx[i, tmpi] = 1
    for i in range(nums):
        print('the given start number:', np.argmax(testx[i]))
        print('the created numbers:   ', model.sample(testx[i]) )

if __name__ == '__main__':
    #x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8]--> y0 = [1, 2, 3, 4, 5, 6, 7, 8, 0],            x1 = [5, 6, 7, 8, 0, 1, 2, 3, 4]--> y1 = [6, 7, 8, 0, 1, 2, 3, 4, 5]
    model = RNN(10, 100, 10)
    state = np.random.random(100)
    epoches = 5
    smooth_loss = 0
    for ll in range(epoches):
        print('epoch i:', ll)
        x, y = getrandomdata(2000)
        for i in range(x.shape[0]):
            h, output = model.forward(x[i])
            loss, state = model.backword(x[i], y[i], h, output, lr=0.001)
            if i == 1:
                smooth_loss = loss
            else:
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
            #print('loss ----  ', smooth_loss)               #if you want to see the cost, you can uncomment this line to observe the cost
    test(7)