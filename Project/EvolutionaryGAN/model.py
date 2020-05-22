import torch.optim as optim
from nets import *

class Model():
    def __init__(self, opt):
        # Create the generator and discriminator
        self.netG = Generator(opt.ngpu, opt.nc, opt.nz, opt.ngf).to(opt.device)
        self.netD = Discriminator(opt.ngpu, opt.nc, opt.ndf).to(opt.device)

        # Handle multi-gpu if desired
        if (opt.device.type == 'cuda') and (opt.ngpu > 1):
            netG = nn.DataParallel(netG, list(range(opt.ngpu)))
            netD = nn.DataParallel(netD, list(range(opt.ngpu)))

        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

         # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
