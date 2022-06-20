import numpy as np

import sys
sys.path.append("/Users/phamtrongdev/Coding/machinelearning")
from com.actvations import sigmoid

class SingleHiddenLayerNet:
    def __init__(self,input_size , output_size, hidden_size = 1):
        print("Create SingleHiddenLayerNetwork",
        f"Input Layers: {input_size},Hidden Layers: {hidden_size}, Output Layers: {output_size}")
        self.parameters  = {"W1":np.array([[0.5,0.5],[1,0.7]]), "W2":np.array([0.5,0.5]), "B1":np.array([0.7,0.5]),"B2":np.array([0.5,0.5])}
    
    def predict(self,inputs):
        # setup input, output vector
        inputs_vector = inputs
        outputs = 0

        # get neuron of each layers
        z1 = sigmoid(np.dot(inputs_vector,self.parameters["W1"]) + self.parameters["B1"]) #neuron in hidden layers
        output = np.dot(z1,self.parameters["W2"] + self.parameters["B2"])

        return output

if __name__ == "__main__":
    Net = SingleHiddenLayerNet(2,1,1)
    print(Net.parameters)

    print(Net.predict([0,1]))