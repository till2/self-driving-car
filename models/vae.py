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

    
class CategoricalDistribution(nn.Module):
    """
    Given a tensor of logits, this module samples from a categorical distribution,
    and computes the straight-through gradient estimator of the sampling operation.
    """
    def __init__(self, num_classes):
        super(CategoricalDistribution, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits):
        # Compute the softmax probabilities
        probs = F.softmax(logits, -1)

        # Sample from the categorical distribution
        m = Categorical(probs)
        sample = F.one_hot(m.sample(), num_classes=self.num_classes)

        # Compute the straight-through gradient estimator
        grad = probs - probs.detach()
        sample = sample + grad
        
        return sample

class VAE(nn.Module):
    def __init__(self, greyscale=True, beta=4):
        super(VAE, self).__init__()

        if greyscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
            
        self.beta = 50 # kl-multiplier (from beta-VAE paper)
        
        self.encoder = nn.Sequential(
            ConvBlock(self.input_channels, 16),
            ResConvBlock(16, 16),
            ConvBlock(16, 16, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(16, 32),
            ResConvBlock(32, 32),
            ConvBlock(32, 32, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 64),
            ResConvBlock(64, 64),
            ConvBlock(64, 64, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 128),
            ResConvBlock(128, 128),
            ConvBlock(128, 128, kernel_size=4, padding=1, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(128, 256),
            ResConvBlock(256, 256),
            ConvBlock(256, 256, kernel_size=4, padding=2, stride=2), #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ResConvBlock(256, 256),
            # ConvBlock(256, 4, kernel_size=4, padding=1, stride=1),
            
            ConvBlock(256, 8, kernel_size=4, padding=1, stride=1),
            nn.Flatten(),
            
        )
        
        self.mu = nn.Linear(8*4*4, 32)
        self.log_var = nn.Linear(8*4*4, 32)

        self.decoder = nn.Sequential(
            TransposeConvBlock(2, 32),
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
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        return mu, log_var
    
    def decode(self, z):
        x = self.decoder(z)
        return x
        
    def forward(self, x):
        # get the distribution of z
        mu, log_var = self.encode(x)
        
        # reparameterization
        std = torch.exp(log_var / 2)
        epsilon = torch.randn_like(log_var)
        
        # sample z
        z = mu + std * epsilon
        
        # reconstruct x
        xhat = self.decode(z.view(-1, 2, 4, 4))
        return xhat, mu, log_var

    def get_loss(self, x, xhat, mu, log_var):
        
        # image reconstruction loss
        reconstruction_loss = F.mse_loss(x, xhat, reduction='sum')
        
        # KL divergence between the latent distribution and the standard normal distribution
        var = torch.exp(log_var)
        kl_divergence = 0.5 * torch.sum(var -log_var -1 +mu.pow(2))
        
        # total loss
        loss = reconstruction_loss + self.beta * kl_divergence
        return loss, reconstruction_loss, kl_divergence

    def save_weights(self):
        if not os.path.exists("weights"):
            os.mkdir("weights")
        torch.save(self.state_dict(), "weights/VAE")
    
    def load_weights(self, path="weights/VAE", eval_mode=True):
        self.load_state_dict(torch.load(path))
        if eval_mode:
            print("Set VAE to evaluation mode.")
            self.eval()
        
    def get_num_params(self):
        return sum(p.numel() for p in vae.parameters() if p.requires_grad)

# vae = VAE(greyscale=True).to(device)

# vae_optim = optim.Adam(
#     vae.parameters(), 
#     lr=3e-4, 
#     weight_decay=1e-5 # l2 regularization
# )

# vae_scheduler = ReduceLROnPlateau(vae_optim, 'min')


# print(vae.get_num_params())