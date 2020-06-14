import argparse
import os
import torch

class Parser():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', help='the path to the root of the dataset folder')
        parser.add_argument('--workers', type=int, required=True, help='the number of worker threads for loading the data with the DataLoader')
        parser.add_argument('--batch_size', type=int, required=True, help='the batch size used in training')
        parser.add_argument('--image_size', type=int, default=64, help='the spatial size of the images used for training')
        parser.add_argument('--nc', type=int, default=3, help='number of color channels in the input images: 3 for RGB and 1 for grayscale')
        parser.add_argument('--nz', type=int, required=True, help='length of latent vector')
        parser.add_argument('--ngf', type=int, required=True, help='relates to the depth of feature maps carried through the generator')
        parser.add_argument('--ndf', type=int, required=True, help='sets the depth of feature maps propagated through the discriminator')
        parser.add_argument('--num_epochs', type=int, required=True, help='number of training epochs to run')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for training')
        parser.add_argument('--beta1', type=float, default=0.5, help='hyperparameter for Adam optimizers')
        parser.add_argument('--beta2', type=float, default=0.999, help='hyperparameter for Adam optimizers')
        parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs available, use 0 for CPU mode')
        parser.add_argument('--toy_type', type=int, help='type of toy dataset, 1 - eight gaussians, 2 - twenty five gaussians, 3 - standard gaussian')
        parser.add_argument('--toy_std', type=float, default=0.2, help='standard deviation of gaussians in mixture')
        parser.add_argument('--toy_scale', type=float, default=1.0, help='scale of unit square for toy dataset')
        parser.add_argument('--toy_len', type=int, default=1000, help='length of toy dataset, technically it is infinite, but used for more generic implementation')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        # set up gpu
        opt.device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

        self.opt = opt
        return self.opt
