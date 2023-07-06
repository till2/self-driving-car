from operator import itemgetter

import torch.nn as nn
import torch.nn.functional as F

from .utils import load_config, to_np


class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, n_layers=None, hidden_dims=None, out_type="linear", weight_init=None):
        super().__init__()

        config = load_config()
        
        self.input_dims = input_dims
        self.hidden_dims = config["mlp_hidden_dims"] if hidden_dims is None else hidden_dims
        self.output_dims = output_dims
        
        self.n_layers = config["mlp_n_layers"] if n_layers is None else n_layers
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
                if out_type == "softmax":
                    activation = nn.Softmax(dim=-1)
                elif out_type == "gaussian":
                    activation = False
                    self.output_mean = nn.Sequential(
                        nn.Linear(layer_in, layer_out),
                        nn.Tanh() # => constrain mean to [-1,1]
                    )
                    self.output_var = nn.Sequential(
                        nn.Linear(layer_in, layer_out),
                        nn.Softplus() # => constrain variance to [0, inf]
                    )
                    
            # define block
            linear = nn.Linear(layer_in, layer_out)
            norm = nn.LayerNorm(layer_out)

            # activation
            if config["activation"].lower() == "silu":
                activation = nn.SiLU(inplace=True) if activation is None else activation
            elif config["activation"].lower() == "elu":
                activation = nn.ELU(inplace=True) if activation is None else activation
            else:
                activation = nn.ReLU(inplace=True) if activation is None else activation
            
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

            # init the weights of the final layer to zeros
            if (i == self.n_layers - 1) and (weight_init == "final_layer_zeros"):
                print(f"Adding zero weight init to the output layer.")
                nn.init.zeros_(linear.weight)
            
            if layer is not None:
                self.layers.add_module(f"layer_{i}", layer)
        
    def forward(self, x):
        x = self.layers(x)
        if self.out_type == "gaussian":
            return self.output_mean(x), self.output_var(x)
        return x