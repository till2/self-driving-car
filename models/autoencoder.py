import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal

import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    """ Use this block to change the number of channels. """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class TransposeConvBlock(nn.Module):
    """ Use this block to change the number of channels and perform a deconvolution
        followed by batchnorm and a relu activation. """
    def __init__(self, in_channels, out_channels):
        super(TransposeConvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class ResConvBlock(nn.Module):
    """ This block needs the same number input and output channels.
        It performs three convolutions with batchnorm, relu 
        and then adds a skip connection. """
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.conv2(x)        
        x += residual
        x = self.relu(x)
        x = self.bn2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, greyscale=True):
        super(Autoencoder, self).__init__()

        if greyscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        
        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, 16),
            ResConvBlock(16, 16),
            nn.Conv2d(16, 16, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(16, 32),
            ResConvBlock(32, 32),
            nn.Conv2d(32, 32, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 64),
            ResConvBlock(64, 64),
            nn.Conv2d(64, 64, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 128),
            ResConvBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(128, 256),
            ResConvBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=4, padding=2, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResConvBlock(256, 256),
            ConvBlock(256, 2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=2),
            TransposeConvBlock(16, 32),
            ResConvBlock(32, 32),
            
            TransposeConvBlock(32, 64),
            ResConvBlock(64, 64),
            
            TransposeConvBlock(64, 128),
            ResConvBlock(128, 128),
            
            TransposeConvBlock(128, 256),
            ResConvBlock(256, 256),
            
            TransposeConvBlock(256, 512),
            ResConvBlock(512, 512),
            
            ResConvBlock(512, 512),
            ConvBlock(512, 32),
            nn.Conv2d(32, self.input_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

        
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        return x
        
    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    def get_loss(self, x, xhat):
        return F.mse_loss(xhat, x)

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        torch.save(self.state_dict(), "weights/AE")
    
    def load_weights(self, path="weights/AE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set AE to evaluation mode.")
            self.eval()
        
    def get_num_params(self):
        return sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

autoencoder = Autoencoder(greyscale=True).to(device)
autoencoder_criterion = nn.MSELoss()

autoencoder_optim = optim.Adam(
    autoencoder.parameters(), 
    lr=3e-4, 
    weight_decay=1e-5 # l2 regularization
)

autoencoder_scheduler = ReduceLROnPlateau(autoencoder_optim, 'min')


print(autoencoder.get_num_params())

# autoencoder.load_weights()
# autoencoder.train();



# rolling_length = 5
# autoencoder_losses_moving_average = (
#     np.convolve(
#         np.array(autoencoder_losses).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# plt.plot(autoencoder_losses_moving_average)



# """ show the observation """
# plt.imshow(torch.permute(batch_tensor[0].cpu(), (1,2,0)), cmap="gray")
# plt.show()

# """ show the reconstruction """
# autoencoder.eval()
# with torch.no_grad():
#     plt.imshow(torch.permute(autoencoder(batch_tensor)[0].cpu(), (1,2,0)), cmap="gray")
#     plt.show()
# autoencoder.train();



# """ latent space exploration """
# n_sliders = 32
# latent_range = (0, 1, 0.01)

# ae2 = Autoencoder(greyscale=True).to(device)
# ae2.load_weights("AE")

# slider_args = {}
# for i in range(n_sliders):
#     slider_args[f"x{i+1}"] = latent_range
    
# @widgets.interact(**slider_args)
# def f(**kwargs):
    
#     slider_values = [
#         kwargs[f"x{i+1}"] for i in range(n_sliders)]
    
#     h = torch.tensor(slider_values, device=device).view(1,2,4,4)
#     with torch.no_grad():
#         plt.imshow(torch.permute(ae2.decoder(h)[0].cpu(), (1,2,0)), cmap="gray")



# create the environment
# toy_env = False
# if toy_env:
#     env = gym.make("CarRacing-v2", render_mode="rgb_array")
# else:
#     exe_path = "/home/till/Desktop/Thesis/donkeycar_sim/DonkeySimLinux/donkey_sim.x86_64"
#     port = 9091
#     config = {
#         "exe_path" : exe_path, 
#         "port" : port 
#     }
#     env = gym.make(
#         "GymV21Environment-v0", 
#         env_id="donkey-minimonaco-track-v0", # donkey-warehouse-v0 
#         make_kwargs={
#             "conf": config
#         })

# n_episodes = 300
# autoencoder_losses = []

# for episode in tqdm(range(n_episodes)):
    
#     # get the initial state
#     obs, info = env.reset()
    
#     # setup a minibatch of x's for training the autoencoder
#     batch_size = 8
#     batch_counter = 0
#     x = transform(obs)
#     batch_tensor = torch.empty((batch_size,) + x.shape, device=device) # B,C,H,W
    
#     # play one episode
#     done = False
#     while not done:
             
#         # add the new x to the batch
#         batch_tensor[batch_counter] = transform(obs)
#         batch_counter += 1
        
#         if batch_counter % batch_size == 0:
#             # reset the batch counter
#             batch_counter = 0
            
#             # autoencoder forward pass with a minibatch
#             xhat = autoencoder(batch_tensor)

#             # get a loss and update the autoencoder
#             autoencoder_loss = autoencoder_criterion(batch_tensor, xhat)
#             autoencoder_optim.zero_grad()
#             autoencoder_loss.backward()
#             autoencoder_optim.step()

#             autoencoder_losses.append(autoencoder_loss.item())
            
#         # choose and execute an action
#         action = env.action_space.sample()
#         next_obs, reward, terminated, truncated, info = env.step(action)        
        
#         # print(next_obs)
#         # env.render()
        
#         done = terminated or truncated
#         obs = next_obs
        
# env.close()