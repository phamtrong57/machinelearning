import numpy as np
import sys
import os
sys.path.append(str(os.getcwd()))
from CNN_DEMO.single_hidden_layer import SingleHiddenLayerNet
def gradient_descent(loss,x):
    gradient = np.ones_like(x)

    it = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])

    while not it.finished:
        index = it.multi_index
        h = 1e-4
        tmp = x[index]

        #fx + h 
        x[index] = tmp + h
        fx1 = loss(x)

        #fx - h
        x[index] = tmp - h
        fx2 = loss(x)

        #calculate the gradient
        gradient[index] = (fx1 - fx2) / (2*h)
        print(f"Gradient of {index} : {gradient}")
        # reset x
        x[index] = tmp
        it.iternext()

if __name__ == "__main__":
    XOR_net  = SingleHiddenLayerNet(2,1,2)

    #init the weight
    XOR_net.init_parameter()
    print(XOR_net.parameters)

    gradient_descent(loss,XOR_net.parameters["W1"])
#calculate the gradient of the loss function
#update the weight
#repeat unitl the loss stop reducing
