import torch 
import torch.nn as nn

if __name__=="__main__":

    inputs = [1,2,3,4]
    inputs[:3] = [2*input for input in inputs[:3] ]  
    print(inputs)