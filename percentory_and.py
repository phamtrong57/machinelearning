import numpy as np
import sys
from lib.file import File
from lib.runtime import Runtime

def AND(x1, x2):
    weight = np.array([0.5,0.5])
    theta = 0.7
    tmp = np.array([x1,x2]) * weight
    if sum(tmp) <= theta:
        return 0
    return 1

def NAND(x1, x2):
    weight = np.array([-0.5,-0.5])
    theta = -0.7
    tmp = np.array([x1,x2]) * weight
    if sum(tmp) <= theta:
        return 0
    return 1


if __name__ == '__main__':
    start =  Runtime.start()
    data = File.readInt(sys.argv[1]) 
    print(*(AND(data[i][0],data[i][1]) for i in range(len(data))))
    print(*(NAND(data[i][0],data[i][1]) for i in range(len(data))))
    stop = Runtime.stop(start)