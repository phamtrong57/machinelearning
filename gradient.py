from mimetypes import init
from re import I
from common import *

def f(x):
    return x[0]**2 + x[1]**2

x = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])

for i in range(3):
    print("-------------",i*100," steps-------------------")
    print(gradient_descent(f,init_x = x,step_num=100*i))