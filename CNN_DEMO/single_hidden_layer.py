from pickletools import int4
import numpy as np

# set current path to CNN_DEMO in order to use functions inside com folder
import sys
sys.path.append("/Users/phamtrongdev/Coding/machinelearning")
from com.actvations import sigmoid
from com.sysargv import get_argv

class SingleHiddenLayerNet:
    def __init__(self,input_size , output_size, hidden_size = 1):
        print("\n","-"*50,sep="")
        print("Create SingleHiddenLayerNetwork",
        f"Input Size: {input_size},Hidden Size: {hidden_size}, Output Size: {output_size}")
        print("-"*50,"\n")

        # init parameters (W1, W2, B1, B2) {Wx: Weights, Bx: Bias}
        self.parameters  = {"W1":np.array([[20,-20],[20,-20]]), "W2":np.array([20,20]), "B1":np.array([-10,30]),"B2":np.array([-30])}
    
    def predict(self,inputs):
        # setup input, output vector
        inputs_vector = inputs
        outputs = 0

        # get neuron of each layers
        z1 = sigmoid(np.dot(inputs_vector,self.parameters["W1"]) + self.parameters["B1"]) #neuron in the hidden layer
        output = sigmoid(np.dot(z1,self.parameters["W2"]) + self.parameters["B2"]) # neuron in the output layer

        return output

if __name__ == "__main__":

    Net = SingleHiddenLayerNet(2,1,1) # Create a new NetWork
    print("Type [ python single_hidden_layer.py 1 0 ] or [ python3 single_hidden_layer.py 1 0 ] to test model")

    input_vector = np.array(get_argv([1,2]),dtype=np.int8) # get input numbers

    print(f"Test Case : {input_vector[0]} {input_vector[1]} -> {Net.predict(input_vector)[0]:.1f}")