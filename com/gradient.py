import numpy as np
def numericial_gradient(f,x):
    
    # init gradient
    gradients = np.zeros_like(x)

    #init h
    h = 1e-4
    #create iter
    it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]

        #fx+h
        x[idx] = tmp + h  
        fxh1 = f(x[idx])

        #fx-h
        x[idx] = tmp - h
        fxh2 = f(x[idx])

        #store gradient 
        gradients[idx] = (fxh1 - fxh2)/(2*h)
        
        #reset x
        x[idx] = tmp
        
        #next value of x
        it.iternext()
    return gradients




