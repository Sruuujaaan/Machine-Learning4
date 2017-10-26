import random
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
random.seed(10)
np.random.seed(10)

# Put test function here, if p=0 then it has some normalized error
# otherwise it returns the actual value of the function
def test_function(x,p=0):
    output = 3*x + 2
    if p==0:
        return output + np.random.normal(0,3,x.shape)
    else: return output

# t determines the type of filter, 0 = linear, 1 = polynomial, 2 = gaussian
def kernel(x1,x2,t,sig=1):
    if t == 0:
        return np.dot(np.transpose(x1),x2)
    elif t == 1:
        return (np.dot(np.transpose(x1),x2)+1)**2
    elif t ==2:
        diff = np.clip( np.linalg.norm(x1-x2), -500, 500 )
        return np.exp(-diff**2/(2*sig**2))

# number of samples generated
N = 50

# Lamda needs to be > 0 even if small for some reason
lamda = 0.1

# Generates the training data for X
x_train = np.reshape(10*np.random.rand(N),(-1,1))
t_train = test_function(x_train)
t_train.reshape(1, -1)

# Calculates K matrix
K = np.zeros((N,N))
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = kernel(x_train[i], x_train[j],2)

# Trains the algorithm on the training data
I = np.identity(K.shape[0])
a = np.dot(inv(K + lamda*I),t_train)

X_plot = np.linspace(0,10,5000)[:,None]
Y_plot = np.zeros((5000,1))

# Calculates k vector and y-value for each test value
k = np.zeros((N,1))
for i in range(0,X_plot.shape[0]):
    for j in range(0,N):
        k[j,0] = kernel(x_train[j],X_plot[i],2)
    Y_plot[i] = np.dot(np.transpose(k),a)

# Plots original line
Y_original = test_function(X_plot,1)
plt.plot(X_plot,Y_original, color='black', lw=2)
plt.plot(X_plot, Y_plot, color='blue', lw=2)
plt.scatter(x_train, t_train, 80, c="g", alpha=1, label="My Title")
plt.show()
