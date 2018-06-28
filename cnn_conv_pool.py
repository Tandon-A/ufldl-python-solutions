import numpy as np 
import scipy.signal

def sigmoid(val):
    return 1/(1+np.exp(-val))

def cnnConvolve(kernel,filters,images,weight,bias):
"""
cnnConvolve Returns the convolution of the features given by W and b with the given images
    
Parameters:
kernel - filter (feature) dimension
filters - number of feature maps
images - large images to convolve with, matrix in the form images(r, c,channel, image number)
weight, bias - weight, bias for features from the sparse autoencoder
weight is of shape (numFilters,filterDim*filterDim)
bias is of shape (numFilters,1)
   
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

Parameters:
poolDim - dimension of pooling region
convolvedFeatures - convolved features to pool (as given by cnnConvolve) convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

Returns:
pooledFeatures - matrix of pooled features in the form pooledFeatures(poolRow, poolCol, featureNum, imageNum)
"""     
    
    num_images = convolvedFeatures.shape[3]
    num_channels = convolvedFeatures.shape[2]
    convolvedDim = convolvedFeatures.shape[0]
    pool_size = convolvedDim/pool_kernel    
    
    pooledFeatures = np.zeros(shape=(pool_size,pool_size,num_channels,num_images))
    
    for row in range(pool_size):
        for col in range(pool_size):
            pool = convolvedFeatures[row*pool_kernel:(row+1)*pool_kernel,col*pool_kernel:(col+1)*pool_kernel,:,:]
            pooledFeatures[row,col,:,:] = np.mean(np.mean(pool,0),0)
   
    return pooledFeatures
   
    
