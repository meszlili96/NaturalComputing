import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
from simdata import SimulatedDistribution, MixtureOfGaussiansDataset

def toy_dataset(opt):
    dataset = MixtureOfGaussiansDataset(SimulatedDistribution(opt.toy_type), opt.toy_std, opt.toy_scale, opt.toy_len)

    # we do not want to shuffle data here, since we have random sampling
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.workers)
    return dataloader


def image_dataset(opt):
    dataroot = os.path.join(os.path.abspath(os.getcwd()), str(opt.dataroot)+'/')
    print(dataroot)
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.image_size),
                                   transforms.CenterCrop(opt.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    return dataset
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.workers)
    return dataloader

def mnist_dataset(opt):
    return DataLoader(
            dset.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )