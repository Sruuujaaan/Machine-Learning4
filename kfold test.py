from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge

# t determines the type of filter, 0 = linear, 1 = polynomial, 2 = gaussian
def kernel(x1,x2,t,sig=1):
    if t == 0:
        return np.dot(np.transpose(x1),x2)
    elif t == 1:
        return (np.dot(np.transpose(x1),x2)+1)**2
    elif t ==2:
        diff = np.clip( np.linalg.norm(x1-x2), -500, 500 )
        return np.exp(-diff**2/(2*sig**2))

# Lamda needs to be > 0 even if small for some reason
lamda = 0.01

# Read in the file
data = pd.ExcelFile('edited_dataset.xlsx')

# Get raw data from the file
df = data.parse('Sheet1')
print df.shape
index = np.uint8(df.shape[0]*.8)
raw_X_train = df.iloc[0:index,0:-1].values
raw_Y_train = df.iloc[0:index,-1].values
final_X_test = df.iloc[index+1:-1,0:-1].values
final_Y_test = df.iloc[index+1:-1,-1].values



# Using kfold validation, split the data into 5 groups and randomize them
# randomstate is just a randomization seed for duplication purposes
sigrange = np.linspace(.01,3,20)
lamdarange = np.linspace(.1,2,20)
for sigma in sigrange:
    for lamda in lamdarange:
        kf = KFold(n_splits=2, shuffle=True,random_state=1)
        for train_index, test_index in kf.split(raw_X_train):
            X_train = raw_X_train[train_index]
            Y_train = raw_Y_train[train_index]
            X_test = raw_X_train[test_index]
            Y_test = raw_Y_train[test_index]

            # Calculates K matrix
            N = X_train.shape[0]
            K = np.zeros((N,N))
            for i in range(0,N):
                for j in range(0,N):
                    K[i,j] = kernel(X_train[i], X_train[j],1,sigma)

            # Trains the algorithm on the training data
            I = np.identity(K.shape[0])
            a = np.dot(inv(K + lamda*I),Y_train)

            # Calculates k vector and y-value for each training value
            k = np.zeros((N,1))
            Y_predict = np.zeros(Y_train.shape)
            for i in range(0,N):
                for j in range(0,N):
                    k[j,0] = kernel(X_train[j],X_train[i],1,sigma)
                Y_predict[i] = np.dot(np.transpose(k),a)

            r2train = r2_score(Y_train, Y_predict)

            # Calculates k vector and y-value for each test value
            k = np.zeros((N,1))
            Y_predict = np.zeros(Y_test.shape)
            for i in range(0,X_test.shape[0]):
                for j in range(0,N):
                    k[j,0] = kernel(X_train[j],X_test[i],2)
                Y_predict[i] = np.dot(np.transpose(k),a)

            r2test = r2_score(Y_test, Y_predict)
            
            if r2test > 0: print "******************************************************"
            print "sigma:", sigma
            print "lamda:", lamda
            print "R-squared of training: ", r2train
            print "R-squared of test: ", r2test
    
    
