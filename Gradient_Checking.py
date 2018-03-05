import numpy as np 


def lin_reg_vec(theta,x,y):
    """
    Function to calculate cost and gradient for given arguments for linear regression model
    """
    val = np.dot(theta,x) - y
    grad = np.transpose(np.dot(x,np.transpose(val)))
    cost = np.sum(val**2)/2
    return cost,grad 


def grad_check(theta,x,y):
    eps = pow(10,-4) 
    grad = np.zeros(theta.shape)
    numf = theta.shape[0] # number of features 
    for i in range(numf):
        evec = np.zeros(theta.shape)
        evec[i] = 1
        cplus,_ = lin_reg_vec(theta + eps*evec,x,y)
        csub,_ = lin_reg_vec(theta - eps*evec,x,y)
        grad[i] = (cplus - csub)/(2*eps)
    return grad
    
data = np.loadtxt("housing.data")  #specify path to data file
data = np.insert(data,0,1,axis=1)
np.random.shuffle(data)

train_data = data[:400,:-1]
train_labels = data[:400,-1]

test_data = data[400:,:-1]
test_labels = data[400:,-1]


train_data = np.transpose(train_data)
test_data = np.transpose(test_data)



theta = np.random.rand(train_data.shape[0])

_,grad_lin_reg = lin_reg_vec(theta,train_data,train_labels)
grad_ch = grad_check(theta,train_data,train_labels)

diff = np.abs(grad_lin_reg - grad_ch)
print (diff)

assert all(np.less(diff,1e-4)) # the difference between the two gradients computed is very less 


