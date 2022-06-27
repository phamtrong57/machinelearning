from pyexpat import model
import numpy as np

# set current path to CNN_DEMO in order to use functions inside com folder
import sys
import os
sys.path.append(str(os.getcwd()))
from com.actvations import sigmoid
from com.sysargv import get_argv
from com.gradient import numericial_gradient
from com.loss_fucntions import mean_square_erorr

class SingleHiddenLayerNet:
    def __init__(self,input_size , output_size, hidden_size = 2):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        print("\n","-"*50,sep="")
        print("create singlehiddenlayernetwork",
        f"input size: {input_size},hidden size: {hidden_size}, output size: {output_size}")
        print("-"*50,"\n")

        # init parameters (w1, w2, b1, b2) {wx: weights, bx: bias}
        self.parameters  = {"W1":np.array([[20,-20],[20,-20]]), "W2":np.array([20,20]), "B1":np.array([-10,30]),"B2":np.array([-30])}
        # self.parameters = {}
        # self.parameters = {"W1":np.ones((input_size,hidden_size)),
        # "W2":np.ones((hidden_size,output_size)),
        # "B1":np.ones(input_size),"B2":np.ones(hidden_size)}
    def set_parameters(self,parameters):
        self.parameters["W1"] = parameters["W1"] 
        self.parameters["W2"] = parameters["W2"] 
        self.parameters["B1"] = parameters["B1"] 
        self.parameters["B2"] = parameters["B2"] 

    def predict(self,inputs):
        # setup input, output vector
        inputs_vector = inputs
        outputs = 0

        # get neuron of each layers
        hidden_layer_neuron = sigmoid(np.dot(inputs_vector,self.parameters["W1"]) + self.parameters["B1"]) #neuron in the hidden layer
        output = sigmoid(np.dot(hidden_layer_neuron,self.parameters["W2"]) + self.parameters["B2"]) # neuron in the output layer

        return output
    
if __name__ == "__main__":

    Net = SingleHiddenLayerNet(2,1,2) # Create a new NetWork with 1 input layer, 1 hidden layer and 1 output layer
    print("Type [ python single_hidden_layer.py 1 0 ] or [ python3 single_hidden_layer.py 1 0 ] to test model")

    input_vector = np.array(get_argv([1,2]),dtype=np.float16) # get input numbers

    print(f"Test Case : {input_vector[0]} {input_vector[1]} -> {Net.predict(input_vector)[0]:.1f}")