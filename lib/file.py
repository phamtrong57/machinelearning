import os 
class File:
    def readInt(self,path):
        intData = []
        try:
            file = open(path)
            for line in file:
                #split and covert str to int
                data = list(map(int,line.split()))
                intData.append(data)
        finally:
            file.close()
        return intData

