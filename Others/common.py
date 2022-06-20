import numpy as np
from sklearn.preprocessing import maxabs_scale
from soupsieve import escape

def sigmoid(x):
    return (1/(1 + np.exp(-x)))
def softmax(a):
    # init max vlue, format exp value of a, and sum of it. 
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)

    # return probability values that were converted from a vector
    y = exp_a / sum_exp_a
    return y 

def cross_entropy_error(predict_output,raw_output):
#    escape of log(0) error that will given a noncountable -integer 
    delta = 1e-7
    error = -np.sum(raw_output*np.log(predict_output + delta))
    return error

def numerical_gradient(f,x):
    """f: function
        x: numpy array"""
    h = 1e-4
    grads = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        #fxh1 
        x[idx] = tmp + h
        fxh1 = f(x)

        #fxh2
        x[idx] = tmp - h
        fxh2 = f(x)

        grads[idx] = ((fxh1 - fxh2)/(2*h))
        x[idx] = tmp

        print("grads:")
        print(grads)
        it.iternext()
    return grads 


def _numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        print("grad:",grad)
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad
def gradient_descent(f,init_x,lr=0.01,step_num = 100):
    x = init_x

    for i in range(step_num):
        gradient = numerical_gradient(f,x)
        x -= lr*gradient        

    return x 