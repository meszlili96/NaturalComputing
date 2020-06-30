from gan import ToyGAN, WGAN
from parser import Parser
from utils import set_seed

def main():
    set_seed()
    results_folder = "../results/"
    # On my laptop it fails if writer is initialised. Don't have time to look into it now
    #writer = SummaryWriter('runs/test')
    opt = Parser().parse()
    # Set up your model here
    gan = ToyGAN(opt)
    #gan = WGAN(opt)
    gan.train(results_folder, None)

if __name__ == '__main__':
    main()
