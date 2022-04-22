import numpy as np

x = np.array([[12,19],[22,44]])
print(x)
print(x.flatten())
x = x.flatten()
print(x[np.array([0,2,3])])
#print elements if they are bigger than 20
print(x[x>20])