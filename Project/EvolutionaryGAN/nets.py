import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F


class ImGenerator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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

class ImDiscriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
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

class MNISTGenerator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTGenerator, self).__init__()
 
        # 1
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
 
        # 2
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
 
        # 3
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 4
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # 5
        out = F.tanh(self.fc4(x))
        return out

class MNISTDiscriminator(nn.Module):
    #https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTDiscriminator, self).__init__()
 
        # 1
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
 
        # 2
        self.fc4 = nn.Linear(hidden_dim, output_size)
 
        # dropout layer 
        self.dropout = nn.Dropout(0.3)
 
 
    def forward(self, x):
        #3 
        x = x.view(-1, 28*28)
        #4 
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # 5
        out = self.fc4(x)
        return out


class CelebaGenerator(nn.Module): #copied from WGAN
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTGenerator, self).__init__()
 
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

class CelebaDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTDiscriminator, self).__init__()
 
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

class PokeGenerator(nn.Module):
    #https://github.com/Zhenye-Na/pokemon-gan/blob/master/Pytorch/model/model.py
    def __init__(self, ngpu, nc, nz, ngf):
        super(PokeGenerator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),

        )

        self.fc_net = nn.Sequential(
            nn.Dropout(),
            nn.Linear(nc * (ngf * ngf), nc),
            nn.ReLU(inplace=True),
            nn.Linear(3, 1)
        )

    def forward(self, input):
        input = self.conv(input)
        #input = input.view(input.size(0), -1)
        #input = self.fc_net(input)
        return input



class PokeDiscriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(PokeDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

        self.fc_net = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192 * (96 * 96), 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 1)
        )

    def forward(self, input):
        input = self.conv(input)
        ##input = input.view(input.size(0), -1)
        #input = self.fc_net(input)
        return input

# custom weights initialization called on netG and netD
def weights_init_celeb(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
