"""
Python solution for Supervised Neural Network Problem. 
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def weight_init(shape):
"""
Function to initialize the weight matrix. 
Arguments: shape: Shape of the weight matrix (W)
"""
    mean = 0
    std = 2/shape[1]
    w = np.random.normal(mean,std,shape)
    w = np.reshape(w,shape)
    return w

def bias_init(shape):
"""
Function to initialize the bias matrix. 
Arguments: shape: Shape of the bias matrix (b)
"""
    b = np.zeros(shape)
    return b

def sigmoid(val):
    val = 1/ (1+ np.exp(-val))
    return val


def softmax(val):
    prob = np.exp(val)
    prob = prob/np.sum(prob,axis=0)
    return prob
    
def forward_pass(x,weights,bias):
"""
Function to compute the activations during forward pass through the NN. 
Arguments: 
x: The input matrix 
weights: List containing the weight matrices. 
bias: List containing the bias matrices. 
"""
    act = []
    a = x
    for i,w,b in  zip(range(len(weights)),weights,bias):
        z = np.matmul(w,a) + b        
        if i != (len(weights)-1):
            a = sigmoid(z)
            act.append(a)
        else:
            prob = softmax(z) #last layer. Softmax gives the final probability of the input belonging to all classes. 
            act.append(prob)
    return act,np.round(prob) #np.round on prob gives the prediction for the input x as one hot encoded label.  
    
def cost_calc(labels,prob):
"""
Using cross entropy cost
Arguments: 
labels: The one-hot encoded label matrix corresponding to the input. 
prob: The Probability matrix as obained by passing the input through the network. 
"""
    c = -np.trace(np.matmul(labels,np.log(prob)))
    return c
    
def error_calc(labels,activations,weights):
"""
Function to calculate the error values at each layer. (in reverse order - last layer error is calculated first)
Arguments:  
labels: One-hot encoded label matrix. 
activations: List containing the activations matrices.
weights: List containing the weight matrices.
"""
    errors = []
    e_out = -(np.transpose(labels)-activations[-1])
    errors.append(e_out)
    e = e_out
    j = len(activations)-2
    while j>=0:
        e = np.matmul(np.transpose(weights[j+1]),e)
        e = np.multiply(e,np.multiply(activations[j],1-activations[j]))
        errors.append(e)
        j = j - 1     
    return errors

def grad_calc(errors,activations,x):
"""
Function to calculate gradients for the weights and bias matrices. 
Arguments:  
errors: List containing error matrices (in reverse order)
activations: List containing activation matrices. 
x: The input matrix. 
"""
    w_grad = []
    b_grad = []
    for j in range(len(errors)):
        if j == 0:
            w_gr = np.matmul(errors[-1],np.transpose(x))
            b_gr = np.sum(errors[-1],axis=1,keepdims=True)
        else:
            w_gr = np.matmul(errors[-(1+j)],np.transpose(activations[j-1]))
            b_gr = np.sum(errors[-(1+j)],axis=1,keepdims=True)
        w_grad.append(w_gr)
        b_grad.append(b_gr)
    return w_grad,b_grad

def reg_cost_calc(weights,decay):
"""
Function to calculate regularization cost. 
Arguments: 
weights: List containing weight matrices.
decay: The regularization constant. 
"""
    w_sum = 0
    for k in range(len(weights)):
        w_sum = w_sum + np.sum(weights[k]*weights[k])
    return (decay/2)*w_sum


def back_prop(x,y,weights,bias,decay,lr_rate):
"""
Function to do back propogation - calculate updated weight and bias matrices
Arguments: 
x: The input matrix. 
y: The one hot encoded label matrix. 
weights: List containing weight matrices. 
bias: List containing bias matrices. 
decay: Regularization constant. 
lr_rate: Learning Rate for the NN. 
"""
    act,pred = forward_pass(np.transpose(x),weights,bias)
    cost = cost_calc(y,act[-1])
    errors = error_calc(y,act,weights)      
    w_grad,b_grad = grad_calc(errors,act,np.transpose(x))
    reg_cost = reg_cost_calc(weights,decay)
    no_of_ex = x.shape[0]
    new_w = []
    new_b = []        
    for w_old,b_old,w_gr,b_gr in zip(weights,bias,w_grad,b_grad):
        w_n = w_old - lr_rate*((w_gr/no_of_ex) + decay*w_old)
        b_n = b_old - lr_rate*(b_gr/no_of_ex)
        new_w.append(w_n)
        new_b.append(b_n)
    #grad_check(x,y,weights,bias,w_grad)
    return new_w,new_b,(cost+reg_cost),pred 

def grad_check(x,y,weights,bias,grad):
"""
Function to do gradient checking
Arguments: 
x: The input matrix. 
y: The output matrix. (One-hot encoded)
weights: List containing weight matrices. 
bias: List containing bias matrices. 
grad: List containing gradient matrices. 
"""
    eps = pow(10,-4) #small number with which parameters will be updated
    for layer in range(int(len(weights)/2)): #doing gradient checking for half the layers and half the neurons
        for row in range(int(weights[layer].shape[0]/2)):
            for col in range(int(weights[layer].shape[1]/2)):
                theta_layer_pos = weights[layer].copy()
                theta_layer_neg = weights[layer].copy()
                theta_layer_pos[row,col] = theta_layer_pos[row,col] + eps
                theta_layer_neg[row,col] = theta_layer_neg[row,col] - eps
                
                theta_pos = weights[:]
                theta_pos[layer] = theta_layer_pos
                theta_neg = weights[:]
                theta_neg[layer] = theta_layer_neg
                
                act_pos,_ = forward_pass(np.transpose(x),theta_pos,bias)
                act_neg,_ = forward_pass(np.transpose(x),theta_neg,bias)
                
                cost_pos = cost_calc(y,act_pos[-1])
                cost_neg = cost_calc(y,act_neg[-1])
                
                grad_norm = (cost_pos - cost_neg)/(2*eps)
                print (layer,row,col,grad_norm,grad[layer][row,col])


def dense_to_one_hot(labels,num_class):
"""
Convert dense labels to one hot encoded.
"""
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels)*num_class
    labels_one_hot = np.zeros((num_labels,num_class))
    labels_one_hot.flat[index_offset+labels.ravel()] = 1
    return labels_one_hot
    
def acc(pred,lab):
"""
Function to calculate accuracy. 
Arguments: 
pred: The predicted matrix - one hot encoded
labels: The label matrix - one hot encoded
"""
    return 100.0 * (np.sum(np.argmax(pred,1) == np.argmax(lab,1))/pred.shape[0])             
                
data = pd.read_csv("mnist.csv")
data = np.array(data)
np.random.shuffle(data)
train_data = data[0:30000,1:]
train_label = data[0:30000,0]
test_data = data[30000:,1:]
test_label = data[30000:,0]
train_data[train_data>0] = 1
test_data[test_data>0] = 1
no_of_classes = 10
test_label = dense_to_one_hot(test_label,no_of_classes)
train_label = dense_to_one_hot(train_label,no_of_classes)
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)


"""
network initialization 
Input Layer - 784 neurons 
Hidden Layer - 256 neurons
Output Layer - 10 neurons -- for MNIST classes
"""

ww= []
bb = []
s1 = 256,784
s2 = 256,1
w = weight_init(s1)
b = bias_init(s2)
ww.append(w)
bb.append(b)
s1 = 10,256
s2 = 10,1
w = weight_init(s1)
b = bias_init(s2)
ww.append(w)
bb.append(b)

decay = 0.0001
lr_rate = 0.05



cost_batches = []
batch_size = 64
max_bs_len = int(30000/batch_size)*batch_size
for epoch in range(5):
      bs = 0
      while bs<(max_bs_len)-batch_size:
          x = train_data[bs:(bs+batch_size)]
          y = train_label[bs:(bs+batch_size)]
          ww_new,bb_new,batch_cost,pred = back_prop(x,y,ww,bb,decay,lr_rate)
          ww = ww_new[:]
          bb = bb_new[:]
          cost_batches.append(batch_cost)
          bs = bs+batch_size
      x = test_data
      y = test_label
      _,test_pred = forward_pass(np.transpose(x),ww,bb)
      acc_cv = acc(test_pred,np.transpose(y))
      print ("accuracy = %r" %(acc_cv))


# plot the training cost 
plt.plot(cost_batches, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
