import numpy as np
import sys
import os
sys.path.append(str(os.getcwd()))

from com.loss_fucntions import cross_entropy_error

t = np.array([0,0,1,0,0,0,0,0,0])
x = np.array([0.1,0.05,0.1,0,0.05,0.1,0.6,0,0])

print(cross_entropy_error(t,x))