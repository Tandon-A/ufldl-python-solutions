import numpy as np 
import scipy.optimize
import time 
import matplotlib.pyplot as plt 
import pandas as pd 
import math


def log_reg(theta,x,y):
    """
    Arguments:
    theta - A vector containing the parameter values to optimize.
    X - The examples stored in a matrix.
        X(i,j) is the i'th coordinate of the j'th example.
    y - The target value for each example.  y(j) is the target for example j.
    
    Basic function to compute cost and gradient for given arguments. 
    """

    no_of_ex = x.shape[1]
    cost = 0
    grad = np.zeros(theta.shape)
    for i in range(no_of_ex):
        val = np.sum(theta[:]*x[:,i])
        val = 1/ (1 + math.exp(-val)) 
        cost = cost + (y[i]*math.log(val)) + (1-y[i])*math.log(1-val)
        grad = grad + x[:,i]*(val-y[i])
    cost = -cost
    return cost,grad
    

    
    
def log_rec_vec(theta,x,y):
    """
    An optimized function to compute cost and gradient for given arguments
    """
    val = np.dot(theta,x) 
    val = 1/(1+np.exp(-val))
    grad = np.transpose(np.dot(x,np.transpose(val - y)))
    cost = -np.sum(y*np.log(val) + (1-y)*np.log(1-val)) 
    return grad,cost    
    
def cost_fun(theta,x,y):
    """
    Function to calculate cost
    """
    val = np.dot(theta,x) 
    val = 1/(1+np.exp(-val))
    cost = -np.sum(y*np.log(val) + (1-y)*np.log(1-val)) 
    return cost
    
def grad_fun(theta,x,y):
    """
    Function to calculate gradient
    """
    val = np.dot(theta,x) 
    val = 1/(1+np.exp(-val))
    grad = np.transpose(np.dot(x,np.transpose(val - y)))
    return grad

def safe_log(x):
    """
    Function to calculate safe_log i.e. replace nan/inf with -1e+4
    """
    l = np.log(x)
    l[np.logical_or(np.isnan(l),np.isinf(l)) ] = -1e+4
    return l 
    
def safe_cost_fun(theta,x,y):
    """
    Function to calculate cost using safe_log
    """
    val = np.dot(theta,x) 
    val = 1/(1+np.exp(-val))
    cost = -np.sum(y*safe_log(val) + (1-y)*safe_log(1-val)) 
    return cost


def accuracy(theta,x,y):
    """
    Function to calculate accuracy of the logistic regression model
    """
    val = np.dot(theta,x)
    val = 1/(1+np.exp(-val))
    correct = np.sum(np.equal(y, val>0.5))
    return correct/y.size

data = pd.read_csv("mnist.csv") #specify the path to csv file of MNIST database
data = np.array(data)
data = np.insert(data,1,1,axis=1)
np.random.shuffle(data)
train = data[0:30000]
test = data[30000:]

#taking data rows with label digit = 0 or label digit = 1
train_data = train[np.logical_or(train[:,0] == 0, train[:,0] == 1), 1:] 
train_label = train[np.logical_or(train[:,0] == 0, train[:,0] == 1), 0]

test_data = test[np.logical_or(test[:,0] == 0, test[:,0] == 1), 1:]
test_label = test[np.logical_or(test[:,0] == 0, test[:,0] == 1), 0]

#normalizing database
train_data[train_data>0] = 1
test_data[test_data>0] = 1
train_data = np.transpose(train_data)
test_data = np.transpose(test_data)

j_hist = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=cost_fun,
    x0=np.random.rand(train_data.shape[0])*0.001,
    args=(train_data, train_label),
    method='L-BFGS-B',
    jac=grad_fun,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: j_hist.append(cost_fun(x, train_data, train_label)),
)
t1 = time.time()
optimal_theta = res.x
print ("Optimization using lbfgs took %r seconds" %(t1-t0))

plt.plot(j_hist, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')



j_hist = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=safe_cost_fun,
    x0=np.random.rand(train_data.shape[0])*0.001,
    args=(train_data, train_label),
    method='bfgs',
    jac=grad_fun,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: j_hist.append(safe_cost_fun(x, train_data, train_label)),
)
t1 = time.time()
optimal_theta = res.x
print ("Optimization using bfgs and safe log took %r seconds" %(t1-t0))

plt.plot(j_hist, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')


print ("training accuracy = %r" %(accuracy(optimal_theta,train_data,train_label)))
print ("testing accuracy = %r" %(accuracy(optimal_theta,test_data,test_label)))

