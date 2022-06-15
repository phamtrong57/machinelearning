import numpy as np

a = np.ones([2,3])
h = 1e-4
gradients = np.ones_like(a)
for i in range(a.ndim):
    tmp = a[i]
    
    a[i] = tmp + h
    fx1 = a[i]**2
    print("fx1:")
    print(fx1)

    a[i] = tmp - h
    fx2 = a[i]**2
    print("fx2:")
    print(fx2)

    gradients[i] = (fx1 - fx2) / (2*h)

print("gradient")
print(gradients)
    