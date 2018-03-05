import numpy as np 
import time 
import scipy.optimize
import matplotlib.pyplot as plt 
import pandas as pd


def softmax(theta,x,y):
    """
    Arguments:
    theta - A matrix containing the parameter values to optimize.
    X - The examples stored in a matrix.
        X(i,j) is the i'th coordinate of the j'th example.
    y - The target value for each example.  y(j) is the target for example j.
    
    Function to calculate cost and gradient for given arguments for softmax model.
    """   
    
    prob = np.exp(np.dot(np.transpose(theta),x))
    prob = prob/np.sum(prob,axis=0)
    c = 0
    for i in range(x.shape[1]):
        c = c + np.log(prob[y[i]][i])
    grad = np.dot(x,np.transpose(prob))
    for i in range(x.shape[1]):
        grad[:,y[i]] = grad[:,y[i]] - x[:,i]
    return -c, grad
    
def cost_fun(theta,x,y):
    """
    Function to calculate cost for softmax model
    """
    theta = theta.reshape((theta.size/no_of_classes,no_of_classes))  #converting the theta vector into a matrix 
    prob = np.exp(np.dot(np.transpose(theta),x))
    prob = prob/np.sum(prob,axis=0)
    c = 0
    for i in range(x.shape[1]):
        c = c + np.log(prob[y[i]][i])
    return -c
    
def grad_fun(theta,x,y):
    """
    Function to calculate gradient for softmax model
    """
    theta = theta.reshape((theta.size/no_of_classes,no_of_classes))    #converting the theta vector into a matrix 
    prob = np.exp(np.dot(np.transpose(theta),x))
    prob = prob/np.sum(prob,axis=0)
    grad = np.dot(x,np.transpose(prob))
    for i in range(x.shape[1]):
        grad[:,y[i]] = grad[:,y[i]] - x[:,i]
    return grad.flatten()
    
def prob_fun(theta,x,y):
    """
    Function to calculate the probability for a digit given the features 
    """
    theta = theta.reshape((theta.size/no_of_classes,no_of_classes)) #converting the theta vector into a matrix   
    prob = np.exp(np.dot(np.transpose(theta),x))
    prob = prob/np.sum(prob,axis=0)
    return prob
    
def accuracy(theta,x,y):
    correct = np.sum(np.argmax(prob_fun(theta,x,y),axis=0) == y)
    return correct/y.size

data = pd.read_csv("mnist.csv")   #specify path to .csv file of MNIST database 
data = np.array(data)
data = np.insert(data,1,1,axis=1)
np.random.shuffle(data)

#keeping first 30k examples for training the softmax regression model and rest for testing 
train_data = data[0:30000,1:]   
train_label = data[0:30000,0]  # the zeroth column is the label column in the mnist.csv file 
test_data = data[30000:,1:]
test_label = data[30000:,0]

train_data = np.transpose(train_data)
train_data[train_data>0] = 1  #normalizing the training data
test_data = np.transpose(test_data)
test_data[test_data>0] = 1    #normalizing the testing data

no_of_classes = np.unique(train_label).size
theta = np.random.rand(train_data.shape[0],no_of_classes)*0.001

j_hist = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=cost_fun,
    x0=theta,
    args=(train_data, train_label),
    method='L-BFGS-B',
    jac=grad_fun,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: j_hist.append(cost_fun(x, train_data, train_label)),
)
t1 = time.time()
optimal_theta = res.x
print ("Optimization took %r seconds" %(t1-t0))

plt.plot(j_hist, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')


print ("training accuracy = %r" %(accuracy(optimal_theta,train_data,train_label)))
print ("testing accuracy = %r" %(accuracy(optimal_theta,test_data,test_label)))

