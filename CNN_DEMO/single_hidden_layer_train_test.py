
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(str(os.getcwd()))
from CNN_DEMO.single_hidden_layer_train import SingleHiddenLayerNet

def train(XOR_net,input_matrix,raw_output_matrix):
    step = 0
    loss_data = np.array([])
    print("LOSS:",XOR_net.loss(input_matrix,raw_output_matrix))

    while XOR_net.loss(input_matrix,raw_output_matrix) > 0.0017: 
        loss = XOR_net.loss(input_matrix,raw_output_matrix)
        # print(f"Predict: {XOR_net.predict(input_matrix).reshape(1,4)} --- RAW: {raw_output_matrix.reshape(1,4)} ---- LOSS: {loss}")
        
        if input_matrix.ndim == 1:
            print(f"Predict: {XOR_net.predict(input_matrix)} --- RAW: {raw_output_matrix} ---- LOSS: {loss}")
        else:
            print(f"Predict: {XOR_net.predict(input_matrix).reshape(1,4)} --- RAW: {raw_output_matrix.reshape(1,4)} ---- LOSS: {loss}")

        #calculate the gradients of the model with current parameters
        gradients = XOR_net.gradient_descent(input_matrix,raw_output_matrix)

        #update the paramters with calculated gradients
        learning_rate = 30
        XOR_net.update_parameters(gradients,learning_rate)

        step += 1 
        loss_data = np.append(loss_data,loss) #store loss in order to plot after

    plt.plot(loss_data)
    print("STEP:",step)
    print("Finally Prameters:\n",XOR_net.parameters)
    print("Input after training:",XOR_net.predict(input_matrix)) 
    plt.show()

if __name__ == "__main__":
    XOR_net  = SingleHiddenLayerNet(2,1,2)

    #init the weight
    XOR_net.init_parameter()

    #calculate the gradient of loss function
    input_matrix_2D = np.array([[1,0],[0,1],[0,0],[1,1]]) #training input data   
    raw_output_matrix_2D = np.array([[1],[1],[0],[0]]) #training output data

    input_matrix_1D = np.array([1,1]) #test input data
    raw_output_matrix_1D = np.array([0]) #test ouput data

    #start training
    train(XOR_net,input_matrix_2D,raw_output_matrix_2D)