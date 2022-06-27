
import numpy as np
import time

# set current path to CNN_DEMO in order to use functions inside com folder
import sys
import os
sys.path.append(str(os.getcwd()))
from com.actvations import sigmoid
from com.gradient import numericial_gradient
from com.loss_fucntions import mean_square_erorr, mean_square_error_batch
import matplotlib.pyplot as plt

class SingleHiddenLayerNet:

    # setup basic data ( input size , output size, hidden size) for training
    def __init__(self,input_size , output_size, hidden_size = 2):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        print("\n","-"*50,sep="")
        print("create singlehiddenlayernetwork",
        f"input size: {input_size},hidden size: {hidden_size}, output size: {output_size}")
        print("-"*50,"\n")

        # init parameters (w1, w2, b1, b2) {wx: weights, bx: bias}
        self.parameters = {}
        self.init_parameter()

    def set_parameters(self,parameters):
        self.parameters["W1"] = parameters["W1"] 
        self.parameters["W2"] = parameters["W2"] 
        self.parameters["B1"] = parameters["B1"] 
        self.parameters["B2"] = parameters["B2"] 

    def init_parameter(self):
        np.random.seed(0)
        self.parameters["W1"] = np.random.normal(1.0,0.5,(self.input_size,self.hidden_size))
        self.parameters["W2"] = np.random.normal(1.0,0.5,(self.hidden_size,self.output_size))
        self.parameters["B1"] = np.random.normal(1.0,1,self.input_size)
        self.parameters["B2"] = np.random.normal(1.0,1,self.output_size)

    def predict(self,inputs):
        # setup input, output vector
        inputs_vector = inputs
        outputs = 0

        # get neuron of each layers
        hidden_layer_neuron = sigmoid(np.dot(inputs_vector,self.parameters["W1"]) + self.parameters["B1"]) #calculate the neuron in the hidden layer with sigmoid func
        output = sigmoid(np.dot(hidden_layer_neuron,self.parameters["W2"]) + self.parameters["B2"]) # calculate the neuron in the output layer with sigmoid func

        return output
    
    def loss(self,input_matrix,raw_output_matrix):
        # get the new output of model with the new parameters
        model_output = self.predict(input_matrix)

        # return error between the output of model and the raw output
        # return mean_square_erorr(model_output,raw_output_matrix) #1D 
        return mean_square_error_batch(model_output,raw_output_matrix)
    
    def gradient_descent(self,input_matrix,raw_output_matrix):
        # get the error with new changed parameters
        loss_parameters_function = lambda parameters : self.loss(input_matrix,raw_output_matrix)

        #calculate the gradients of each parametes
        gradient = {}
        gradient["W1"] = numericial_gradient(loss_parameters_function,self.parameters["W1"]) # The weights between the input and hidden layer
        gradient["W2"] = numericial_gradient(loss_parameters_function,self.parameters["W2"]) # The weights between the hidden layer and the output layer
        gradient["B1"] = numericial_gradient(loss_parameters_function,self.parameters["B1"]) # The bias between the input and hidden layer
        gradient["B2"] = numericial_gradient(loss_parameters_function,self.parameters["B2"]) # The bias between the hidden layer and the output layer
        return gradient

    def update_parameters(self,gradient,learning_rate=0.1):
        #update the parameters => prameters = parameters - learning rate * gradients of the parameters
        self.parameters["W1"] -= learning_rate * gradient["W1"]
        self.parameters["W2"] -= learning_rate * gradient["W2"]
        self.parameters["B1"] -= learning_rate * gradient["B1"]
        self.parameters["B2"] -= learning_rate * gradient["B2"]
