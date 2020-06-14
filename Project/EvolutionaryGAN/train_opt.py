from gan import ToyGAN, CelebGAN
from parser import Parser
from train import train_gan


def main():
    opt = Parser().parse()
    # Set up your model here
    gan = ToyGAN(opt)
    train_gan(gan, opt)

if __name__ == '__main__':
    main()
