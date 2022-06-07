# from common import softmax, cross_entropy_error, numerical_gradient
import numpy as np
# import matplotlib.pyplot as plt

# def f(x):
#     return x[0]**2 + x[1]**2


# def loss(t,y):
#     z = softmax(y)
#     loss = cross_entropy_error(z,t)
#     return loss

# t = np.random.rand(2)
# print(loss(f,t,y))

a = np.random.rand(2,3)

for i, x in enumerate(a):
    print(i,x)