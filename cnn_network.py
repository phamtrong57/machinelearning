from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
from common import cross_entropy_error, numerical_gradient, softmax, sigmoid,_numerical_gradient

class CNN_Network:
    def __init__(self, input_size, hidden_size, output_size):
        
        #set up cnn layer size
        self.input_layer_size = input_size
        self.hidden_layer_size = hidden_size
        self.output_layer_size = output_size

        #create network_parameters in order to store weights and bias of each layer
        self.network_parameters = {"W1":[],"W2":[],"W3":[],"b1":[],"b2":[],"b3":[]}
        self.init_weights_and_bias()

    def set_input_vector(self,x):
        self.input_vector = x
    
    def init_weights_and_bias(self):
        w1 = np.random.randn(self.input_layer_size,self.hidden_layer_size)
        w2 = np.random.randn(self.hidden_layer_size,self.output_layer_size)
        b1 = np.ones(self.hidden_layer_size) 
        b2 = np.ones(self.output_layer_size) 

        self.network_parameters["W1"] = w1 
        self.network_parameters["W2"] = w2
        self.network_parameters["b1"] = b1
        self.network_parameters["b2"] = b2


    def predict(self,x):
        w1,w2 = self.network_parameters["W1"], self.network_parameters["W2"]
        b1,b2 = self.network_parameters["b1"], self.network_parameters["b2"] 

        #calculate values of hidden layer and output layer
        hiddenlayer_vector = np.dot(x,w1) + b1

        #activate function sigmoid
        activated_hiddenlayer_vector = sigmoid(hiddenlayer_vector)

        outputlayer_vector = np.dot(activated_hiddenlayer_vector,w2) + b2

        #covert to probability with softmax function
        softmax_outputlayer_vector = softmax(outputlayer_vector)

        return softmax_outputlayer_vector

    def loss(self,x,raw_output):
        predict_output = self.predict(x)
        return cross_entropy_error(predict_output,raw_output)
    
    def gradient(self,x,raw_output):
        loss_W = lambda W: self.loss(x,raw_output)
        gradient_vectors =  {}

        #calculate gradient of each parameter with cross entropy error method
        gradient_vectors["W1"] = numerical_gradient(loss_W,self.network_parameters["W1"]) #get gradient of W1
        gradient_vectors["W2"] = numerical_gradient(loss_W,self.network_parameters["W2"]) #get gradient of W2
        gradient_vectors["b1"] = numerical_gradient(loss_W,self.network_parameters["b1"]) #get gradient of b2
        gradient_vectors["b2"] = numerical_gradient(loss_W,self.network_parameters["b2"]) #get gradient of b2

        return gradient_vectors
    def _gradient(self,x,raw_output):
        loss_W = lambda W: self.loss(x,raw_output)
        gradient_vectors =  {}

        #calculate gradient of each parameter with cross entropy error method
        gradient_vectors["W1"] = _numerical_gradient(loss_W,self.network_parameters["W1"]) #get gradient of W1
        gradient_vectors["W2"] = _numerical_gradient(loss_W,self.network_parameters["W2"]) #get gradient of W2
        gradient_vectors["b1"] = _numerical_gradient(loss_W,self.network_parameters["b1"]) #get gradient of b2
        gradient_vectors["b2"] = _numerical_gradient(loss_W,self.network_parameters["b2"]) #get gradient of b2

        return gradient_vectors