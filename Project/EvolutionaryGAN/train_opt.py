from gan import ToyGAN, CelebGAN
from parser import Parser
from torch.utils.tensorboard import SummaryWriter


def main():
    results_folder = "../results/"
    # On my laptop it fails if writer is initialised. Don't have time to look into it now
    #writer = SummaryWriter('runs/test')
    opt = Parser().parse()
    # Set up your model here
    gan = ToyGAN(opt)
    gan.train(results_folder, None)

if __name__ == '__main__':
    main()
