import numpy as np
def step_function(x,delta):
    return 1 if x > delta else 0

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    return (np.exp(x)/np.sum(np.exp(x)))
