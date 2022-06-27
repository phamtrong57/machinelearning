import numpy as np
class Example:
    def __init__(self):
        self.x = {"X1":np.array([000]),"X2":np.array([222])}
    def update_parameter(self,new_x):
        new_x[0] = np.array([111])

ex = Example()
print(ex.x)
ex.update_parameter(ex.x["X1"])
print(ex.x)