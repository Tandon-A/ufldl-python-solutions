import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize
import time 

def lin_reg(theta,x,y):
    """
    Arguments:
    theta - A vector containing the parameter values to optimize.
    X - The examples stored in a matrix.
        X(i,j) is the i'th coordinate of the j'th example.
    y - The target value for each example.  y(j) is the target for example j.
    
    A basic function computing cost and gradient for the given arguments.
    """

    no_of_ex = x.shape[1]
    grad = np.zeros(theta.shape)
    cost = 0
    for i in range(no_of_ex):
        val = (np.sum(theta[:]*x[:,i]) - y[i])
        cost = cost + pow(val,2)
        grad = grad + x[:,i]*val
    cost = cost/2
    return cost,grad
    
    
def lin_reg_vec(theta,x,y):
    """
    An optimized function to compute the cost and gradient for the given arguments
    """
    val = np.dot(theta,x) - y
    grad = np.transpose(np.dot(x,np.transpose(val)))
    cost = np.sum(val**2)/2
    return cost,grad     
    
def cost_fun(theta,x,y):
    """
    Function to calculate cost 
    """
    val = np.dot(theta,x) - y
    cost = np.sum(val**2)/2
    return cost
    
def grad_fun(theta,x,y):
    """
    Function to calculate gradient
    """
    val = np.dot(theta,x) - y
    grad = np.transpose(np.dot(x,np.transpose(val)))
    return grad

def rms_error(theta,x,y):
    """
    Function to calculate RMS error
    """
    val = np.dot(theta,x) - y
    error = np.sqrt(np.mean(val ** 2))
    return error

data = np.loadtxt("housing.data")  #specify the path of data file
data = np.insert(data,0,1,axis=1)
np.random.shuffle(data)

train_data = data[:400,:-1]
train_labels = data[:400,-1]

test_data = data[400:,:-1]
test_labels = data[400:,-1]


train_data = np.transpose(train_data)
test_data = np.transpose(test_data)

j_hist = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=cost_fun,
    x0=np.random.rand(train_data.shape[0]),
    args=(train_data, train_labels),
    method='bfgs',
    jac=grad_fun,
    options={'maxiter': 200, 'disp': True},
    callback=lambda x: j_hist.append(cost_fun(x, train_data, train_labels)),
)
t1 = time.time()
optimal_theta = res.x
print ("Optimization took %r seconds" %(t1-t0))

plt.plot(j_hist, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')


plt.figure(figsize=(10, 8))
plt.scatter(np.arange(test_labels.size), sorted(test_labels), c='b', edgecolor='None', alpha=0.5, label='actual')
plt.scatter(np.arange(test_labels.size), sorted(optimal_theta.dot(test_data)), c='g', edgecolor='None', alpha=0.5, label='predicted')
plt.legend(loc='upper left')
plt.ylabel('House price ($1000s)')
plt.xlabel('House #')


print ("training rms error = %r" %(rms_error(optimal_theta,train_data,train_labels)))
print ("testing rms error = %r" %(rms_error(optimal_theta,test_data,test_labels)))
