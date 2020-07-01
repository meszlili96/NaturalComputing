import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


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



class PokeGenerator(nn.Module):
    #https://github.com/Zhenye-Na/pokemon-gan/blob/master/Pytorch/model/model.py
    def __init__(self, ngpu, nc, nz, ngf):
        super(PokeGenerator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf*8, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=3, stride=1, padding=2),
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
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*2, out_channels=ndf*4, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*4, out_channels=ndf*8, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=ndf*8, out_channels=1, kernel_size=3, stride=1, padding=2),
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
