from gan import ToyGAN
from egan import ToyEGAN
from parser import Parser
from utils import set_seed

def main():
    set_seed()
    results_folder = "../results/"
    # On my laptop it fails if writer is initialised. Don't have time to look into it now
    #writer = SummaryWriter('runs/test')
    opt = Parser().parse()
    # Set up your model here
    if opt.gan_type == "GAN":
        gan = ToyGAN(opt)
    elif opt.gan_type == "EGAN":
        gan = ToyEGAN(opt)
    # TODO: add WGAN
    gan.train(results_folder, None)

if __name__ == '__main__':
    main()
