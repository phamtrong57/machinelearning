import numpy as np
import sys
sys.path.append("/Users/phamtrongdev/Coding/machinelearning")
from com.actvations import sigmoid
from com.sysargv import get_argv

print(f"{float(sigmoid(1)):.3f}")
print(get_argv([1,2]))