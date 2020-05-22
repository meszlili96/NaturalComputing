import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset

def create_dataset(opt):
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
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.workers)
    return dataloader
