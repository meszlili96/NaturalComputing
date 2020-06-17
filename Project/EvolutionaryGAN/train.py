import torch.nn.parallel
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from gan import ToyGAN, CelebGAN, ToyOptions, CelebOptions

from torch.utils.tensorboard import SummaryWriter

"""To define a new GAN create generator and discriminator NNs with architecture you need and define
   a subclass of GAN object in gan.py file to return these NNs. Implement other methods appropriately.
   Then, create a shell script with GAN parameters. Change the model to use in main function.
"""

def main():
    results_folder = "results/"
    writer = SummaryWriter('runs/test')
    # Change the default parameters if needed
    opt = ToyOptions()
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder, writer)

if __name__ == '__main__':
    main()
