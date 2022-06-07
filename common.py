import numpy as np
from sklearn.preprocessing import maxabs_scale
from soupsieve import escape

def softmax(a):
    # init max vlue, format exp value of a, and sum of it. 
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)

    # return probability values that were converted from a vector
    y = exp_a / sum_exp_a
    return y 

def cross_entropy_error(y,t):
#    escape of log(0) error that will given a noncountable -integer 
    delta = 1e-7
    error = -np.sum(t*np.log(y + delta))
    return error

def numerical_gradient(f,x):
    delta = 1e-7

    grad = np.ones_like(x)

    for i in range(x.size):
        # get current x value
        tmp = x[i]

        #f(x+h)
        x[i] = tmp + delta
        fxh1 = f(x)

        #f(x-h)
        x[i] = tmp - delta 
        fxh2 = f(x)

        #reset x 
        x[i] = tmp

        #restore grad
        grad[i] = (fxh1 - fxh2)/ 2*delta
    return grad