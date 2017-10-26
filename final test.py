from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from kernel_Reg import kernel_Reg

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

#################### LINEAR KERNEL ############################
lamda = .01
linear = kernel_Reg(raw_X_train,raw_Y_train,lamda)
linear.train(0)

# Predict the training values and calculate r2
Y_predict = linear.predict(raw_X_train)
r2train = r2_score(raw_Y_train, Y_predict)

# Predict test values
Y_predict = linear.predict(final_X_test)
r2test = r2_score(final_Y_test, Y_predict)

# Display the values
print "Linear Kernel"
print "lamda: ", lamda
print "R2 train: ", r2train
print "R2 test:  ", r2test, '\n'

#################### POLYNOMIAL KERNEL ############################
lamda = 5
linear = kernel_Reg(raw_X_train,raw_Y_train, lamda)
linear.train(1)

# Predict the training values and calculate r2
Y_predict = linear.predict(raw_X_train)
r2train = r2_score(raw_Y_train, Y_predict)

# Predict test values
Y_predict = linear.predict(final_X_test)
r2test = r2_score(final_Y_test, Y_predict)

# Display the values
print "Polynomial Kernel"
print "lamda: ", lamda
print "R2 train: ", r2train
print "R2 test:  ", r2test, '\n'


#################### POLYNOMIAL KERNEL ############################
lamda = .01
sigma = 5
linear = kernel_Reg(raw_X_train,raw_Y_train, lamda, sigma)
linear.train(2)

# Predict the training values and calculate r2
Y_predict = linear.predict(raw_X_train)
r2train = r2_score(raw_Y_train, Y_predict)

# Predict test values
Y_predict = linear.predict(final_X_test)
r2test = r2_score(final_Y_test, Y_predict)

# Display the values
print "Gaussian Kernel"
print "lamda: ", lamda
print "sigma: ", sigma
print "R2 train: ", r2train
print "R2 test:  ", r2test, '\n'
