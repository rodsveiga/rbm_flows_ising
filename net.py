import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''
    def __init__(self, 
                 layers_size, 
                 out_size,
                 params_list= None):
        
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k+1]))
            
        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)
        
        m_aux = 0
        
        for m in self.layers:
            
            if params_list is None:
                nn.init.normal_(m.weight, mean= 0, std= 1/np.sqrt(100*len(layers_size)))
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1
                
        if params_list is None:
            nn.init.normal_(self.out.weight, mean= 0, std= 1/np.sqrt(100*len(layers_size)))
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]
       
    def forward(self, x):
        
        for layer in self.layers:
                       
            # Activation function
            x = torch.tanh(layer(x))
        
        # Last layer: softmax
        output= F.softmax(self.out(x), dim=1)
        
        return output