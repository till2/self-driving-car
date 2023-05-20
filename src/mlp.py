from operator import itemgetter

import torch.nn as nn
import torch.nn.functional as F

from .utils import load_config, to_np


class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, out_type="linear"):
        super(MLP, self).__init__()

        config = load_config()
        
        self.input_dims = input_dims
        self.hidden_dims = itemgetter("mlp_hidden_dims")(config)
        self.output_dims = output_dims
        
        self.n_layers = itemgetter("mlp_n_layers")(config)
        self.layers = nn.Sequential()
        self.out_type = out_type
        
        for i in range(self.n_layers):
            layer, layer_in, layer_out, activation = None, None, None, None
            
            # input layer
            if i == 0:
                layer_in = self.input_dims
            else:
                layer_in = self.hidden_dims
            
            # output layer
            if i == self.n_layers - 1:
                layer_out = self.output_dims
            else:
                layer_out = self.hidden_dims
            
            # change output layer type
            if i == self.n_layers - 1:
                if out_type == "linear":
                    activation = False
                if out_type == "sigmoid":
                    activation = nn.Sigmoid()
                elif out_type == "gaussian":
                    activation = False
                    self.output_mean = nn.Sequential(
                        nn.Linear(layer_in, layer_out),
                        nn.Tanh() # => [-1,1]
                    )
                    self.output_var = nn.Sequential(
                        nn.Linear(layer_in, layer_out),
                        nn.Sigmoid() # => [0, 1]
                    )
                    
            # define block
            linear = nn.Linear(layer_in, layer_out)
            norm = nn.LayerNorm(layer_out)
            activation = nn.SiLU(inplace=True) if activation is None else activation
            
            # add layer to the network
            if i == self.n_layers - 1 and out_type == "gaussian":
                pass
            else:
                layer = nn.Sequential(
                    linear,
                )
            # for hidden layers add norm and activation
            if i < self.n_layers - 1:
                layer.add_module("1", norm)
                layer.add_module("2", activation)
                
            # for output layer add specified activation
            elif activation:
                layer.add_module("2", activation)
            
            if layer is not None:
                self.layers.add_module(f"layer_{i}", layer)
        
    def forward(self, x):
        x = self.layers(x)
        if self.out_type == "gaussian":
            return self.output_mean(x), self.output_var(x)

        return x