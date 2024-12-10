# https://www.researchgate.net/publication/361521310_Connections_Between_Numerical_Algorithms_for_PDEs_and_Neural_Networks/figures
import torch 
import torch.nn as nn 

class PeronaMalik(nn.Module): 
    def __init__(self, l=1.0): 
        super(PeronaMalik, self).__init__() 
        self.l = l
        
    def forward(self, x): 
        return x/(1+(x**2/self.l**2))

class Charbonnier(nn.Module): 
    def __init__(self, l=1.0): 
        super(Charbonnier, self).__init__() 
        self.l = l
        
    def forward(self, x): 
        return x/torch.sqrt((1+(x**2/self.l**2)))

