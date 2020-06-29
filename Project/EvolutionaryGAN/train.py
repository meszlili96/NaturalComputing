from utils import  set_seed
from gan import ToyGAN, CelebGAN, ToyOptions, CelebOptions

from torch.utils.tensorboard import SummaryWriter

"""To define a new GAN create generator and discriminator NNs with architecture you need and define
   a subclass of GAN object in gan.py file to return these NNs. Implement other methods appropriately.
   Then, create a shell script with GAN parameters. Change the model to use in main function.
"""

def main():
    set_seed()
    # 8 gaussians
    # MinMax
    results_folder = "8 gauss minmax/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.02
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)

    # Heuristic
    results_folder = "8 gauss heuristic/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.02
    opt.g_loss = 2
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)

    # Least squares
    results_folder = "8 gauss least squares/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.02
    opt.g_loss = 3
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)

    
    # 25 gaussians
    # MinMax
    results_folder = "25 gauss minmax/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.05
    opt.toy_type = 2
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)

    # Heuristic
    results_folder = "25 gauss heuristic/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.05
    opt.toy_type = 2
    opt.g_loss = 2
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)

    # Least squares
    results_folder = "25 gauss least squares/"
    # Change the default parameters if needed
    opt = ToyOptions()
    opt.toy_std = 0.05
    opt.toy_type = 2
    opt.g_loss = 3
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder)



if __name__ == '__main__':
    main()
