#import dependacies
from scipy import signal
from scipy import misc
import numpy as np 
from numpy import zeros

#function to obtain individual rows
def conv_row(X):
    return X.flatten()

#unfolding the matrix as described in article
def unfold(mat_X, k):
    n, m = mat_X.shape[0:2]
    X = zeros(((n-k+1)*(m-k+1), k**2))
    row = 0
    for i in range(n-k+1):
        for j in range(m-k+1):
        #Converting to rows
            X[row, :] = conv_row(mat_X[i:i+k, j:j+k])
            row = row + 1
    
    return X

# defining the kernel
W = np.array([[1,2,3],[4,5,6],[7,8,9]], np.float32) #for convolutions, flip by 180 degrees

#define input matrix
X = np.random.randn(5,5)

n, m = X.shape[0:2]
k = W.shape[0]

#The function will be tested against this
Y = signal.correlate2d(X, W, mode = 'valid')

X_unfold = unfold(X, k)
W_flat = W.flatten()

#Correlation as Matrix Multiplication
Y_corr = np.matmul(X_unfold, W_flat)
Y_corr = Y_corr.reshape((n-k+1, m-k+1))

#Print to confirm whether the Matrix Multiplication gives the expected result
print(Y_corr)
print(Y)
