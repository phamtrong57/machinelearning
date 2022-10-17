import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import test, display_parameters

if __name__ == "__main__":
    model_name = "ADaGrad"
    # load model from Backup folder
    for hidden_size in (2**p for p in range(0,6)):

        model = torch.load(os.path.join(os.getcwd(),"Backup",f"{model_name}_{hidden_size}.pth"))
        
        # create data for testing
        input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)

        # test model
        test(model,input_data)

        # display parameters of model
        display_parameters(model,"Pretrained")
    plt.show()