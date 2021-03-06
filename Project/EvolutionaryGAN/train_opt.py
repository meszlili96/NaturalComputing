from gan import *
from egan import *
from parser import Parser
from utils import set_seed

def main():
    set_seed()
    results_folder = "../../results/"
    # On my laptop it fails if writer is initialised. Don't have time to look into it now
    #writer = SummaryWriter('runs/test')
    opt = Parser().parse()
    # Set up your model here
    if opt.gan_type == "ToyGAN":
        gan = ToyGAN(opt)
    elif opt.gan_type == "ToyEGAN":
        gan = ToyEGAN(opt)
    elif opt.gan_type == "ToyWGAN":
        gan = ToyWGAN(opt)
    elif opt.gan_type == "MNISTGAN":
        gan = MNISTGAN(opt)
    elif opt.gan_type == "CelebWGAN":
        gan = CelebWGAN(opt)
    elif opt.gan_type == "CelebGAN":
        gan = CelebGAN(opt)
    gan.train(results_folder, None)

if __name__ == '__main__':
    main()
