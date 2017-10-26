import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge

# Makes random numbers consistent between tests
random.seed(10)
np.random.seed(10)

# Put test function here, if p=0 then it has some normalized error
# otherwise it returns the actual value of the function
def test_function(x,p=0):
    output = 3*x + 2
    if p==0:
        return output + np.random.normal(0,3,x.shape)
    else: return output

# Number of samples generated
N=50

# Create N random X values in range 0 - 10 to train the kernel regression
x_train = 10*np.random.rand(N)
x_train = np.reshape(x_train,(x_train.shape[0],1))

# Y values are normally distributed around y = 3x+2
# with a std dev of 3
y_train = 3*x_train + 2 + np.random.normal(0,3,x_train.shape)
y_train.reshape(1, -1)

# kernel can be: ‘linear’, ‘poly’, ‘rbf’
# 'linear' = linear kernel (duh)
# 'poly' = polynomail kernel
# 'rbf' = gaussian kernel
kr = KernelRidge(alpha=1.0, kernel='linear', gamma=0.1)
kr.fit(x_train,y_train)

# Predicts 1000's of values to create a smooth curve
X_plot = np.linspace(0,10,5000)[:,None]
y_kr = kr.predict(X_plot)

# Plots original line
Y_plot = test_function(X_plot,1)

# Plots the results
# Black line is the actual line
# Blue line is the estimated line
plt.plot(X_plot, y_kr, color='blue', lw=2)
plt.scatter(x_train, y_train, 80, c="g", alpha=1, label="My Title")
plt.plot(X_plot, Y_plot, color='black', lw=2)
plt.show()
