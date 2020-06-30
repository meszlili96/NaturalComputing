import torch.nn as nn
import numpy as np

class DCGANGenerator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(DCGANGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# custom weights initialization called on netG and netD
def weights_init_DCGAN(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ToyDiscriminator(nn.Module):
    def __init__(self, hidden_dim=512):
        super(ToyDiscriminator, self).__init__()
        # Number of input features is 2, since we work with 2d gaussians
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(1)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()
        # Sigmoid activation is for output binary classification layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, sample):
        x = self.relu(self.layer_1(sample))
        x = self.relu(self.batch_norm(self.layer_2(x)))
        x = self.relu(self.layer_3(x))
        output = self.sigmoid(self.layer_out(x))
        return output

class ToyGenerator(nn.Module):
    def __init__(self, hidden_dim=512):
        super(ToyGenerator, self).__init__()
        # Number of input features is 2, since our noise is 2D
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 2)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()

    def forward(self, noise):
        x = self.relu(self.layer_1(noise))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        output = self.layer_out(x)
        out_shape = output.shape
        # Reshape the output to discriminator input format
        return output.reshape((out_shape[0], 1, out_shape[1]))

# The function to initialize NN weights
# Recursively applied to the layers of the passed module
# m - nn.Module object
def weighs_init_toy(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class WassersteinGenerator(nn.Module):
    def __init__(self, nz, nc, image_size):
        super(WassersteinGenerator, self).__init__()
        
        self.img_shape = (nc, image_size, image_size)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(nz, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

# Compared to a simple GAN, WGAN doesn't use sigmoid
class WassersteinDiscriminator(nn.Module):
    def __init__(self, nc, image_size):
        super(WassersteinDiscriminator, self).__init__()
        
        self.img_shape = (nc, image_size, image_size)
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity