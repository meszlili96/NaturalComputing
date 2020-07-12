import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNISTGenerator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTGenerator, self).__init__()
 
        # 1
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
 
        # 2
        self.fc4 = nn.Linear(hidden_dim*4, output_size)
 
        # 3
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 4
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # 5
        out = F.tanh(self.fc4(x))
        return out

class MNISTDiscriminator(nn.Module):
    #https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTDiscriminator, self).__init__()
 
        # 1
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
 
        # 2
        self.fc4 = nn.Linear(hidden_dim, output_size)
 
        # dropout layer 
        self.dropout = nn.Dropout(0.3)
 
 
    def forward(self, x):
        #3 
        x = x.view(-1, 28*28)
        #4 
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        # 5
        out = self.fc4(x)
        return out


class CelebaGenerator(nn.Module): #copied from WGAN
    def __init__(self, nz, nc, image_size):
        super(CelebaGenerator, self).__init__()
 
        self.img_shape = (nc, image_size, image_size)
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(nz, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class CelebaDiscriminator(nn.Module):
    def __init__(self, nc, image_size):
        super(CelebaDiscriminator, self).__init__()
 
        self.img_shape = (nc, image_size, image_size)
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

# custom weights initialization called on netG and netD
def weights_init_celeb(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
