from cnn_network import CNN_Network
import numpy as np


network = CNN_Network(input_size=4,hidden_size=2,output_size=2)

input_vector = np.array([[2,3,4,5],[2,3,4,6]])
print("---------gardient--------------")
gradient = network.gradient(input_vector,[0,1])
print(gradient)
print("---------_gradient-------------")
_gradient = network._gradient(input_vector,[0,1])
print(_gradient)