import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

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
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTGenerator, self).__init__()
 
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
    def __init__(self, input_size, hidden_dim, output_size):
        super(MNISTDiscriminator, self).__init__()
 
        # Number of input features is 2, since our noise is 2D
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 2)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()

    def forward(self, noise):
        x = self.relu(self.layer_1(noise))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        output = self.layer_out(x)
        out_shape = output.shape
        # Reshape the output to discriminator input format
        return output.reshape((out_shape[0], 1, out_shape[1]))

# custom weights initialization called on netG and netD
def weights_init_DCGAN(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ToyDiscriminator(nn.Module):
    def __init__(self, hidden_dim=100):
        super(ToyDiscriminator, self).__init__()
        # Number of input features is 2, since we work with 2d gaussians
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(1)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()
        # Sigmoid activation is for output binary classification layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, sample):
        x = self.relu(self.layer_1(sample))
        x = self.relu(self.batch_norm(self.layer_2(x)))
        x = self.relu(self.layer_3(x))
        output = self.sigmoid(self.layer_out(x))
        return output

class ToyGenerator(nn.Module):
    def __init__(self, hidden_dim=100):
        super(ToyGenerator, self).__init__()
        # Number of input features is 2, since our noise is 2D
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 2)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()

    def forward(self, noise):
        x = self.relu(self.layer_1(noise))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        output = self.layer_out(x)
        out_shape = output.shape
        # Reshape the output to discriminator input format
        return output.reshape((out_shape[0], 1, out_shape[1]))


# The function to initialize NN weights
# Recursively applied to the layers of the passed module
# m - nn.Module object
def weighs_init_toy(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
        
class WassersteinToyDiscriminator(nn.Module):
    def __init__(self, hidden_dim=100):
        super(WassersteinToyDiscriminator, self).__init__()
        # Number of input features is 2, since we work with 2d gaussians
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 1)
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(1)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()

    def forward(self, sample):
        x = self.relu(self.layer_1(sample))
        x = self.relu(self.batch_norm(self.layer_2(x)))
        x = self.relu(self.layer_3(x))
        output = self.layer_out(x)
        return output

class WassersteinToyGenerator(nn.Module):
    def __init__(self, hidden_dim=100):
        super(WassersteinToyGenerator, self).__init__()
        # Number of input features is 2, since our noise is 2D
        # Define 3 dense layers with the same number of hidden units
        self.layer_1 = nn.Linear(2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.layer_out = nn.Linear(hidden_dim, 2)
        # Relu activation is for hidden layers
        self.relu = nn.ReLU()

    def forward(self, noise):
        x = self.relu(self.layer_1(noise))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        output = self.layer_out(x)
        out_shape = output.shape
        # Reshape the output to discriminator input format
        return output.reshape((out_shape[0], 1, out_shape[1]))

class WassersteinGenerator(nn.Module):
    def __init__(self, nz, nc, image_size):
        super(WassersteinGenerator, self).__init__()
        
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

# Compared to a simple GAN, WGAN doesn't use sigmoid
class WassersteinDiscriminator(nn.Module):
    def __init__(self, nc, image_size):
        super(WassersteinDiscriminator, self).__init__()
        
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