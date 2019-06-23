import numpy as np 
import pandas as pd 

#load MNIST data 
data = pd.read_csv('./sample_data/mnist_test.csv')
data = data.drop('7',axis=1) #dropping the label column 
data = np.array(data)[0:1000].T
print (data.shape)

#PCA by eigenvector decomposition 
#mean normalization of dataset features
data = (data - np.mean(data,axis=1).reshape(-1,1))
covar = np.cov(data)  #covar = np.matmul(data,data.T)/data.shape[1] 
eig_value,eig_vector = np.linalg.eig(covar)

#rotated dataset (dataset in a new linear basis)
new_dataset = np.matmul(eig_vector.T,data)
print (np.cov(new_dataset))  #should be a diagonal matrix 

#Calculating the number of principal components to keep such that more than 95% variance is retained
sum_eigs = np.sum(eig_value)
run_sum = 0
for i in range(len(eig_value)):
  run_sum += eig_value[i]  
  if run_sum/sum_eigs > 0.95:
    print (i)
    break

#retained dataset    
PC = eig_vector.T[:i+1,:]
rot_dataset = np.matmul(PC,data)
print (rot_dataset.shape)    

#PCA by SVD

#load mnist data 
data = pd.read_csv('./sample_data/mnist_test.csv')
data = data.drop('7',axis=1) #dropping the label column
data = np.array(data)[0:1000].T
print (data.shape)

#svd of covariance matrix of data
u,s,vt = np.linalg.svd(np.cov(data))
print (u.shape,s.shape,vt.shape)

#rotated dataset (dataset in a new linear basis)
new_dataset = np.matmul(u.T,data)
print (np.cov(new_dataset))

#Calculating the number of principal components to keep such that more than 95% variance is retained
sum_eigs = np.sum(s)
run_sum = 0
for i in range(len(s)):
  run_sum += s[i]  
  if run_sum/sum_eigs > 0.95:
    print (i)
    break

#retained dataset      
PC = u.T[:i+1,:]
rot_dataset = np.matmul(PC,data)
print (rot_dataset.shape)    

eps = 1e-5
#PCA Whitening 
PCAwhite = new_dataset / np.sqrt(s.reshape(-1,1)  + eps)
#XCA Whitening
ZCAWhite = np.matmul(u,PCAwhite)

