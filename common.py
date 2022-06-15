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
    """f: function
        x: numpy array"""
    h = 1e-4
    
    gradients = np.ones_like(x)

    for i in range(x.ndim):
        tmp = x[i]

        #fx+h 
        x[i] = tmp + h
        fx1 = f(x[i])

        #fx-h
        x[i] = tmp -h
        fx2 = f(x[i])

        #rest x
        x[i] = tmp

        #get gradient of x
        gradient = (fx1 - fx2) / (2*h)

        #store gradient of x 
        gradients[i] = gradient
    return gradients 

def gradient_descent(f,init_x,lr=0.01,step_num = 100):
    x = init_x

    for i in range(step_num):
        gradient = numerical_gradient(f,x)
        x -= lr*gradient        

    return x 