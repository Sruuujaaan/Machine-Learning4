import numpy as np
from numpy.linalg import inv

class kernel_Reg(object):

    # Inputs dataset, lamda, sigma for gaussian, and type of kernel you want implemented
    def __init__(self, inputDat, outputDat, Lamda=.01,Sigma=1,Type=0):
        self.inputDat = inputDat
        self.outputDat = outputDat
        self.Lamda = Lamda
        self.Sigma = Sigma
        self.Type = Type
        self.N = inputDat.shape[0]

    # Returns the kernel value given x and x'
    # t determines the type of filter, 0 = linear, 1 = polynomial, 2 = gaussian
    def kernel(self,x1,x2,t,sig=1):
        t = self.Type
        if t == 0:
            return np.dot(np.transpose(x1),x2)
        elif t == 1:
            return (np.dot(np.transpose(x1),x2)+1)**2
        elif t ==2:
            diff = np.clip( np.linalg.norm(x1-x2), -500, 500 )
            return np.exp(-diff**2/(2*sig**2))

    # Trains the dataset
    def train(self,t):
        self.Type = t
        # Calculates K matrix
        K = np.zeros((self.N,self.N))
        for i in range(0,self.N):
            for j in range(0,self.N):
                K[i,j] = self.kernel(self.inputDat[i], self.inputDat[j],self.Type, self.Sigma)

        # Trains the algorithm on the training data
        I = np.identity(K.shape[0])
        self.a = np.dot(inv(K + self.Lamda*I),self.outputDat)

    # Predicts the output given an input array
    def predict(self,predictX):
        predictY = np.zeros((predictX.shape[0],1))
        
        k = np.zeros((self.N,1))
        for i in range(0,predictX.shape[0]):
            for j in range(0,self.N):
                k[j,0] = self.kernel(self.inputDat[j],predictX[i],self.Type, self.Sigma)
            predictY[i] = np.dot(np.transpose(k),self.a)

        return predictY




