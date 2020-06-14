from abc import ABCMeta, abstractmethod
from enum import Enum
import torch.optim as optim
from nets import *
from data import *
from gen_losses import *
from simdata import ToyGenerator, ToyDiscriminator, weighs_init_toy, save_sample
from discr_loss import DiscriminatorLoss

# Model abstract class is a formal protocol, which defines the objects used in GAN training
# It could be not very Pythonish, but it is the best way I came up with so far
class Model():
    __metaclass__ = ABCMeta
    def __init__(self, opt):
        self.opt = opt
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

        # Handle multi-gpu if desired
        if (opt.device.type == 'cuda') and (opt.ngpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(opt.ngpu)))
            self.generator = nn.DataParallel(self.generator, list(range(opt.ngpu)))

        # Apply the weights_init function to randomly initialize all weights as specified
        self.discriminator.apply(self.weights_init_func())
        self.generator.apply(self.weights_init_func())

        # Lists to keep track of progress
        self.d_losses = []
        self.g_losses = []

        # Initialize Discriminator and generator loss functions
        self.d_loss = self.create_d_loss()
        self.g_loss = self.create_g_loss()

        # Setup Adam optimizers for both G and D
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        self.dataset = self.create_dataset()

    @abstractmethod
    def create_discriminator(self):
        raise NotImplementedError

    @abstractmethod
    def create_generator(self):
        raise NotImplementedError

    @abstractmethod
    def create_d_loss(self):
        raise NotImplementedError

    @abstractmethod
    def create_g_loss(self):
        raise NotImplementedError

    @abstractmethod
    def weights_init_func(self):
        raise NotImplementedError

    @abstractmethod
    def create_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def save_gen_sample_func(self, path):
        raise NotImplementedError


class ToyModel(Model):
    def create_discriminator(self):
        return ToyDiscriminator()

    def create_generator(self):
        return ToyGenerator()

    def create_d_loss(self):
        return DiscriminatorLoss()

    def create_g_loss(self):
        return Minmax()

    def weights_init_func(self):
        return weighs_init_toy

    def create_dataset(self):
        return toy_dataset(self.opt)

    def save_gen_sample(self, sample, path):
        save_sample(sample, path)


class CelebModel(Model):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []

    def create_discriminator(self):
        return Discriminator(self.opt.ngpu, self.opt.nc, self.opt.ndf).to(self.opt.device)

    def create_generator(self):
        return Generator(self.opt.ngpu, self.opt.nc, self.opt.nz, self.opt.ngf).to(self.opt.device)

    def create_d_loss(self):
        return DiscriminatorLoss()

    def create_g_loss(self):
        return Minmax()

    def weights_init_func(self):
        return weights_init_celeb

    def create_dataset(self):
        return image_dataset(self.opt)

    def save_gen_sample(self, sample, path):
        plt.figure()
        plt.imshow(sample)
        plt.savefig(path)
        plt.close()