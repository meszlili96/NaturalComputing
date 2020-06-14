import torch.optim as optim
from nets import *
from data import *
from gen_losses import *
from simdata import ToyGenerator, ToyDiscriminator, weighs_init_toy, save_sample
from discr_loss import DiscriminatorLoss


class Options():
    def __init__(self, ngpu=0):
        self.num_epochs = 32
        self.ngpu = 0
        self.lr = 1e-03
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.g_loss = 1
        self.batch_size = 100
        self.workers = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")


class ToyOptions(Options):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.toy_type = 1
        self.toy_std = 0.2
        self.toy_scale = 1.0
        self.toy_len = 100000


class CelebOptions(Options):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.nc = 3
        self.ndf = 64
        self.ngf = 64
        self.nz = 100
        self.image_size = 64
        self.dataroot = "celeba"


"""
GAN abstract class is a formal protocol, which defines the objects used in GAN training
Properties:
    discriminator - a discriminator NN
    generator - a generator NN
    d_loss - a discriminator loss function
    g_loss - a generator loss function
    d_optimizer - a discriminator optimizer
    g_optimizer - a generator optimizer. Note: both use Adam with the parameters from options now.
                  To change this the class should be redesigned
    dataset - a dataset to train on
    d_losses - a Python array to track discriminator loss statistics
    g_losses - a Python array to track generator loss statistics
It could be not very Pythonish, but it is the best way I came up with so far
"""
class GAN():
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


    # Subclasses should implement the methods below and return the corresponding objects
    """Defines and returns discriminator NN, torch.nn.Module object
    """
    @abstractmethod
    def create_discriminator(self):
        raise NotImplementedError

    """Defines and returns generator NN, torch.nn.Module object
    """
    @abstractmethod
    def create_generator(self):
        raise NotImplementedError

    """Defines and returns discriminator loss function, torch.nn.Module object
    """
    def create_d_loss(self):
        return DiscriminatorLoss()

    """Defines and returns generator loss function, torch.nn.Module object
    """
    @abstractmethod
    def create_g_loss(self):
        if self.opt.g_loss == 1:
            return Minmax()
        elif self.opt.g_loss == 2:
            return Heuristic()
        elif self.opt.g_loss == 3:
            return LeastSquares()
        else:
            raise ValueError

    """Defines and returns function that initializes NN weights
    """
    @abstractmethod
    def weights_init_func(self):
        raise NotImplementedError

    """Defines and a dataset to train on
    """
    @abstractmethod
    def create_dataset(self):
        raise NotImplementedError

    """Saves a generated sample at a specified path
       Parameters:
          path - where to save a sample
    """
    @abstractmethod
    def save_gen_sample_func(self, path):
        raise NotImplementedError


class ToyGAN(GAN):
    def create_discriminator(self):
        return ToyDiscriminator()

    def create_generator(self):
        return ToyGenerator()

    def weights_init_func(self):
        return weighs_init_toy

    def create_dataset(self):
        return toy_dataset(self.opt)

    def save_gen_sample(self, sample, path):
        save_sample(sample, path)


class CelebGAN(GAN):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []

    def create_discriminator(self):
        return Discriminator(self.opt.ngpu, self.opt.nc, self.opt.ndf).to(self.opt.device)

    def create_generator(self):
        return Generator(self.opt.ngpu, self.opt.nc, self.opt.nz, self.opt.ngf).to(self.opt.device)

    def weights_init_func(self):
        return weights_init_celeb

    def create_dataset(self):
        return image_dataset(self.opt)

    def save_gen_sample(self, sample, path):
        plt.figure()
        plt.imshow(sample)
        plt.savefig(path)
        plt.close()