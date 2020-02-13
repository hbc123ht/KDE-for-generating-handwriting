from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from mnist import MNIST
from random import seed
from random import randint
import math
def loaddata():
    mnist = MNIST('/Users/cong/Downloads/MNIST')
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)
    data = [[0] * 784] * 1000
    data = np.array(data)
    datal = [[0] * 794] * 1000
    datal = np.array(datal)
    for i in range(1000):
        for j in range(784):
            data[i,j] = x_train[i,j]
            datal[i,j] = x_train[i,j]
            datal[i][y_train[i] + 784] = 1000000
    return (data,datal,y_train,x_test,x_train)
def mer(data,x):
    datal = [0] * 794
    for i in range(784):
        datal[i] = data[i]
    datal[x + 784] = 1000000
    return datal
class KDE(object):
    def __init__(self,data,K):
        self.data = np.array(data)
        self.K = K
    def dot(self,x,y):
        return np.linalg.norm(x - y)

    def dis(self,x):
        num = 0.0
        for i in range(self.data.shape[0]):
            num += 1 / (self.data.shape[0] * self.K * np.sqrt(2 * np.pi)) * np.exp( -(self.dot(self.data[i],x)) ** 2 / (2 * (self.K ** 2) * 100))
        return num

def addnoise(x):
    tmp = np.random.rand(784) * 50
    return x + tmp
def generate(num):
    result = [0] * 784
    global c
    while (1 == 1):
        c += 1
        X = addnoise(x_test[c])
        tmp = KDE2.dis(mer(X, num)) / KDE1.dis(X)
        print(tmp)
        if (tmp >= 0.2):
            result = X
            break
    return result

def plotnum():
    seed(1)
    fig = plt.figure(figsize = (8,4))
    for i in range(1,11):
        re = generate(randint(0,9)) #generating new data with the given random numbers from 0 to 9
        fig.add_subplot(2, 5, i)
        plt.imshow(re.reshape(28,28))
    plt.show()

def predict(x):
    re = 0
    tmp = 0
    X = addnoise(x)
    for i in range(0,9):
        tmpp = KDE2.dis(mer(X, i)) / KDE1.dis(X)
        print(tmpp)
        if (tmpp >= 0.2):
            re = re + 1
    return (re <= 1)
def test():
    re = 0
    for i in range(1001,1200):
        re += predict(x_train[i])
    return re

data,datal,y_train,x_test,x_train = loaddata()
KDE1 = KDE(data,37.64935806792467)#these number are the chosen bandwidths
KDE2 = KDE(datal,35.2970730273065)
c = 0

plotnum()
# in order to reduce the runtime, i already predicted the bandwidth with the code following:
"""
params = {'bandwidth': np.logspace(-1, 2, 200)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)
K = grid.best_estimator_.bandwidth
"""
