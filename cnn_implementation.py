import numpy as np 
import scipy.signal 
import pandas as pd 
import matplotlib.pyplot as plt 

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cnnConvolve(kernel,filters,images,weight,bias):
"""
cnnConvolve Returns the convolution of the features given by W and b with the given images

Argumentss:
kernel - filter (feature) dimension
filters - number of feature maps
images - large images to convolve with, matrix in the form images(r, c,channel, image number)
weight, bias - weight, bias for features from the sparse autoencoder
bias is of shape (numFilters,1)
weight is of shape (numFilters,kernel*kernel*channels) 

Returns:
convolvedFeatures - matrix of convolved features in the form convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
"""
    num_images = images.shape[3]
    image_size = images.shape[1]
    num_channels = images.shape[2]
    convDim = image_size - kernel + 1
    
    convolvedFeatures = np.zeros(shape=(convDim,convDim,filters,num_images))
    
    for imagenum in range(num_images):
        for filterNum in range(filters):
            
            convolvedImage = np.zeros(shape=(convDim,convDim))
            
            for channel in range(num_channels):
                feature_mat = weight[filterNum,(kernel*kernel)*channel:(kernel*kernel)*(channel+1)].reshape(kernel,kernel)
                feature = np.flipud(np.fliplr(feature_mat))
                img = images[:,:,channel,imagenum]
                
                convolvedImage = convolvedImage + scipy.signal.convolve2d(img,feature,mode='valid')
                
            convolvedImage = sigmoid(convolvedImage + bias[filterNum])
            
            convolvedFeatures[:,:,filterNum,imagenum] = convolvedImage
    
    return convolvedFeatures


def cnnPool(pool_kernel,convolvedFeatures):
"""    
cnnPool Pools the given convolved features

Arguments:
poolDim - dimension of pooling region
convolvedFeatures - convolved features to pool (as given by cnnConvolve) convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

Returns:
pooledFeatures - matrix of pooled features in the form pooledFeatures(poolRow, poolCol, featureNum, imageNum)
"""     
    
    num_images = convolvedFeatures.shape[3]
    num_channels = convolvedFeatures.shape[2]
    convolvedDim = convolvedFeatures.shape[0]
    pool_size = int(convolvedDim/pool_kernel)    
    
    pooledFeatures = np.zeros(shape=(pool_size,pool_size,num_channels,num_images))
    
    for row in range(pool_size):
        for col in range(pool_size):
            pool = convolvedFeatures[row*pool_kernel:(row+1)*pool_kernel,col*pool_kernel:(col+1)*pool_kernel,:,:]
            pooledFeatures[row,col,:,:] = np.mean(np.mean(pool,0),0)
   
    return pooledFeatures
    
    
def cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses):
"""
Initializes cnn Parameters - weight and bias matrices
	
Arguments:
imageDim: Height/Width of image
filterDim: Kernel size of convolution filter
numFilters: Number of convolution filters
poolDim: Kernel size of pool operation
numClasses: number of output classes

Returns the weight and bias matrices
"""
  weights= []    
  bias = []
  w_shape = numFilters,filterDim*filterDim    
  w = np.random.normal(0,0.1,w_shape)
  w = np.reshape(w,w_shape)
  weights.append(w)
    
  outDim = imageDim - filterDim + 1 # dimension of convolved image
  outDim = int(outDim/poolDim)
  hiddenSize = int((outDim*outDim)*numFilters)

  r  = np.sqrt(6) / np.sqrt(numClasses+hiddenSize+1);
  w = np.random.normal(0.0,r,(numClasses,hiddenSize))
  weights.append(w)
  b = np.zeros((numFilters,1))
  bias.append(b)
  b = np.zeros((numClasses,1))    
  bias.append(b)
    
  return weights,bias
    
def softmax(val):
    prob = np.exp(val)
    prob = prob/np.sum(prob,axis=0)
    return prob

def forward_pass(x,weights,bias,imageDim,filterDim,poolDim,numFilters,numImages):
"""
Computes activations obtained on passing input through the network

Arguments:
x: Input Matrix of shape (imageDim,imageDim,numChannels,numImages)
weights: List containing weight matrices
bias: List containing bias matrices 
imageDim: Height/Width of input image
filterDim: kernel size of convolution filter 
poolDim: kernel size of pool filter 
numFilters: number of convolution filters
numImages: number of images to be passed through the network

Returns list having activations of the network and the predictions on the input 
"""
  act = []
    
  convolved_act = cnnConvolve(filterDim,numFilters,x,weights[0],bias[0])
  act.append(convolved_act)
  pool_act = cnnPool(poolDim,convolved_act)
  outDim = imageDim - filterDim + 1 # dimension of convolved image
  outDim = int(outDim/poolDim)
  hiddenSize = int((outDim*outDim)*numFilters)
  pool_act = pool_act.reshape((hiddenSize,numImages))
  act.append(pool_act)    
    
  inp_softmax = np.matmul(weights[1],pool_act) + bias[1]
  prob = softmax(inp_softmax)
  act.append(prob)
  return act,np.round(prob)


def cost_calc(labels,prob):
"""
Using cross entropy cost

Arguments: 
labels: The one-hot encoded label matrix corresponding to the input. 
prob: The Probability matrix as obained by passing the input through the network. 
"""
c = -np.trace(np.matmul(labels,np.log(prob)))
return c
    
def error_calc(labels,activations,weights,convDim,poolDim,outputDim,numFilters,numImages):
"""
Function to calculate the error values at each layer. (in reverse order - last year error is calculated first).
    
Arguments:  
labels: One-hot encoded label matrix. 
activations: List containing the activations matrices.
weights: List containing the weight matrices.
convDim: Dimension of array after applpying convolution filter on image.
poolDim: Dimension of pool kernel.
outputDim: Dimension after applying pool operation.
numFilters: number of convolution filters.
numImages: number of images in the input.

Returns the list containing the error matrices.	
"""
 errors = []
 e_out = -(np.transpose(labels)-activations[-1])
 errors.append(e_out)
 e_out = np.matmul(np.transpose(weights[-1]),e_out)
 e_out = e_out.reshape((outputDim,outputDim,numFilters,numImages))
 err_conv = np.zeros((convDim,convDim,numFilters,numImages))
 conv_act = activations[0]
 for imageNum in range(numImages):
     for filterNum in range(numFilters):
         delta = e_out[:,:,filterNum,imageNum]
         deltaPool = (1/(poolDim*poolDim)) * np.kron(delta,np.ones((poolDim,poolDim))) #upsampling pooling error
         err = np.multiply(deltaPool,np.multiply(conv_act[:,:,filterNum,imageNum],(1-conv_act[:,:,filterNum,imageNum])))
         err_conv[:,:,filterNum,imageNum] = err
 errors.append(err_conv)
 return errors
    
def grad_calc(errors,activations,x,weights,bias,numFilters,numImages,filterDim):
"""
Function to calculate gradients for the weights and bias matrices. 

Arguments:  
errors: List containing error matrices (in reverse order)
activations: List containing activation matrices. 
x: The input matrix. 
weights: List containing weight matrices. 
bias: List containing bias matrices. 
numFilters: number of convolution filters. 
numImages: number of images in input
filterDim: kernel size of convolution filter
	
Returns lists containing weight gradients and bias gradients.
"""
    w_grad = []
    b_grad = []
    w_dense = np.matmul(errors[0],np.transpose(activations[1]))
    b_dense = np.sum(errors[0],axis=1,keepdims=True)
    w_conv = np.zeros(weights[0].shape)
    b_conv = np.zeros(bias[0].shape)
    err_conv = errors[-1]
    for filterNum in range(numFilters):
        for imageNum in range(numImages):
            for channel in range(x.shape[2]):
              img = x[:,:,channel,imageNum]
              filter_err = err_conv[:,:,filterNum,imageNum]
              b_conv[filterNum] = b_conv[filterNum] + np.sum(np.sum(filter_err))
              filter_err = np.flipud(np.fliplr(filter_err))
              err_filt = scipy.signal.convolve2d(img,filter_err,mode='valid')
              w_conv[filterNum,:] = w_conv[filterNum,:] + np.reshape(err_filt,(1,filterDim*filterDim))
    w_grad.append(w_conv)
    w_grad.append(w_dense)
    b_grad.append(b_conv)
    b_grad.append(b_dense)        
    return w_grad,b_grad
    
def back_prop(x,y,weights,bias,lr_rate,imageDim,filterDim,poolDim,numImages,numFilters):
"""
Function to do back propogation - calculate updated weight and bias matrices
	
Arguments:
x: The input matrix. 
y: The one hot encoded label matrix. 
weights: List containing weight matrices. 
bias: List containing bias matrices. 
lr_rate: Learning Rate for the NN. 
imageDim: Height/Width of image
filterDim: kernel size of convolution filter 
poolDim: kernel size of pool operation
numImages: number of training images
numFilters: number of convolution filters

Returns lists containing updates for weight and bias matrices. 
"""
    
    act,pred = forward_pass(x,weights,bias,imageDim,filterDim,poolDim,numFilters,numImages)
    cost = cost_calc(y,act[-1])
    outDim = imageDim - filterDim + 1 
    outDim = int(outDim/poolDim)
    errors = error_calc(y,act,weights,(imageDim-filterDim+1),poolDim,outDim,numFilters,numImages)     
    w_grad,b_grad = grad_calc(errors,act,x,weights,bias,numFilters,numImages,filterDim)
    new_w = []
    new_b = []        
    for w_old,b_old,w_gr,b_gr in zip(weights,bias,w_grad,b_grad):
        w_n = w_old - lr_rate*(w_gr/numImages) 
        b_n = b_old - lr_rate*(b_gr/numImages)
        new_w.append(w_n)
        new_b.append(b_n)
    #grad_check(x,y,weights,bias,w_grad,imageDim,filterDim,poolDim,numFilters,numImages)
    return new_w,new_b,cost,pred 

def grad_check(x,y,weights,bias,grad,imageDim,filterDim,poolDim,numFilters,numImages):
"""
Function to do gradient checking. 

Arguments: 
x: The input matrix. 
y: The output matrix. (One-hot encoded)
weights: List containing weight matrices. 
bias: List containing bias matrices. 
grad: List containing gradient matrices. 
imageDim: Height/Width of image
filterDim: kernel size of convolution filter 
poolDim: kernel size of pool operation
numImages: number of training images
numFilters: number of convolution filters 
"""
    eps = pow(10,-4) #small number with which parameters will be updated
    #checking gradinet for conv layer 
    theta_conv = weights[0]
    for filterNum in range(numFilters):
        for i in range(filterDim*filterDim):
            theta_pos = theta_conv.copy()
            theta_neg = theta_conv.copy()
            theta_pos[filterNum,i] = theta_pos[filterNum,i] + eps
            theta_neg[filterNum,i] = theta_neg[filterNum,i] - eps
            
            theta_conv_pos = weights[:]
            theta_conv_pos[0] = theta_pos
            theta_conv_neg = weights[:]
            theta_conv_neg = theta_neg
            
            act_pos,_ = forward_pass(x,theta_conv_pos,bias,imageDim,filterDim,poolDim,numFilters,numImages)
            act_neg,_ = forward_pass(x,theta_conv_neg,bias,imageDim,filterDim,poolDim,numFilters,numImages)
            
            cost_pos = cost_calc(y,act_pos[-1])
            cost_neg = cost_calc(y,act_neg[-1])
            
            grad_norm = (cost_pos - cost_neg)/(2*eps)
            print (grad_norm,grad[0][filterNum][i])
            

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
            
            
            


# Configuration
imageDim = 28
numClasses = 10  # Number of classes (MNIST images fall into 10 classes)
filterDim = 9    # Filter size for conv layer
numFilters = 20  # Number of filters for conv layer
poolDim = 2      # Pooling dimension, (should divide imageDim-filterDim+1)




data = pd.read_csv("mnist.csv")
data = np.array(data)
np.random.shuffle(data)
train_data = data[0:30000,1:]
train_label = data[0:30000,0]
test_data = data[30000:,1:]
test_label = data[30000:,0]
train_data[train_data>0] = 1
test_data[test_data>0] = 1
test_label = dense_to_one_hot(test_label,numClasses)
train_label = dense_to_one_hot(train_label,numClasses)
train_data = np.reshape(np.transpose(np.array(train_data)),(imageDim,imageDim,1,train_label.shape[0]))
train_label = np.array(train_label)
test_data = np.reshape(np.transpose(np.array(test_data)),(imageDim,imageDim,1,test_label.shape[0]))
test_label = np.array(test_label)


#Initialize Parameters
weights,bias = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses)

print ("starting")
lr_rate = 0.001
cost_batches = []
batch_size = 64
max_bs_len = int(train_label.shape[0]/batch_size)*batch_size
max_epoch = 15
for epoch in range(max_epoch):
      bs = 0
      while bs<(max_bs_len)-batch_size:
          x = train_data[:,:,:,bs:(bs+batch_size)]
          y = train_label[bs:(bs+batch_size)]
          ww_new,bb_new,batch_cost,pred = back_prop(x,y,weights,bias,lr_rate,imageDim,filterDim,poolDim,batch_size,numFilters)
          weights = ww_new[:]
          bias = bb_new[:]
          cost_batches.append(batch_cost)
          bs = bs+batch_size
      x = test_data
      y = test_label
      _,test_pred = forward_pass(x,weights,bias,imageDim,filterDim,poolDim,numFilters,test_label.shape[0])
      acc_cv = acc(test_pred,np.transpose(y))
      print ("accuracy = %r" %(acc_cv))
      lr_rate = lr_rate/2


# plot the training cost plot
plt.plot(cost_batches, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
