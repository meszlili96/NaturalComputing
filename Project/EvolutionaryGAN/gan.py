import os
import torch
import torch.optim as optim
from torch.autograd import Variable

from nets import *
from data import *
from utils import *
from gen_losses import *
from simdata import extract_xy
from discr_loss import DiscriminatorLoss
from torchvision import datasets
import datetime

class Options():
    def __init__(self, ngpu=0):
        self.num_epochs = 100
        self.ngpu = 1
        self.lr = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.g_loss = 1
        self.batch_size = 64
        self.workers = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")


class ToyOptions(Options):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.toy_type = 1
        self.toy_std = 0.05
        self.toy_scale = 2.0
        self.toy_len = 500*self.batch_size


class CelebOptions(Options):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.nc = 3
        self.nz = 100
        self.image_size = 64
        self.dataroot = "celeba"

class MNISTGANOptions(Options):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.dataroot = "data/MNIST"
        self.input_size = 784
        self.d_output_size = 1
        self.d_hidden_size = 32
        self.nz = 100
        self.g_output_size = 784
        self.g_hidden_size = 32
"""
GAN abstract class is a formal protocol, which defines the objects used in GAN training
Properties:
    discriminator - a discriminator NN
    generator - a generator NN
    d_loss - a discriminator loss function
    g_loss - a generator loss function
    d_optimizer - a discriminator optimizer
    g_optimizer - a generator optimizer. Note: both use Adam with the parameters from options now.
                  To change this the class should be redesigned
    dataset - a dataset to train on
    d_losses - a Python array to track discriminator loss statistics
    g_losses - a Python array to track generator loss statistics
It could be not very Pythonish, but it is the best way I came up with so far
"""
class GAN():
    __metaclass__ = ABCMeta
    def __init__(self, opt):
        self.opt = opt
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

        # Handle multi-gpu if desired
        if (opt.device.type == 'cuda') and (opt.ngpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(opt.ngpu)))
            self.generator = nn.DataParallel(self.generator, list(range(opt.ngpu)))

        # Apply the weights_init function to randomly initialize all weights as specified
        self.discriminator.apply(self.weights_init_func())
        self.generator.apply(self.weights_init_func())

        # Lists to keep track of progress
        self.d_losses = []
        self.g_losses = []

        # Initialize Discriminator and generator loss functions
        self.d_loss = self.create_d_loss()
        self.g_loss = self.create_g_loss()

        # Setup Adam optimizers for both G and D
        self.d_optimizer = self.create_d_optimizer(opt)
        self.g_optimizer = self.create_g_optimizer(opt)

        self.dataset = self.create_dataset()
        self.data_loader = self.create_data_loader()


    # Subclasses should implement the methods below and return the corresponding objects
    """Defines and returns discriminator NN, torch.nn.Module object
    """
    @abstractmethod
    def create_discriminator(self):
        raise NotImplementedError

    """Defines and returns generator NN, torch.nn.Module object
    """
    @abstractmethod
    def create_generator(self):
        raise NotImplementedError

    """Defines and returns discriminator loss function, torch.nn.Module object
    """
    def create_d_loss(self):
        return DiscriminatorLoss()

    """Defines and returns generator loss function, torch.nn.Module object
    """
    def create_g_loss(self):
        if self.opt.g_loss == 1:
            return Minmax()
        elif self.opt.g_loss == 2:
            return Heuristic()
        elif self.opt.g_loss == 3:
            return LeastSquares()
        else:
            raise ValueError

    """Defines the optimizer for the discriminator
    """
    def create_d_optimizer(self,opt):
        return optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    
    """Defines the optimizer for the generator
    """
    def create_g_optimizer(self,opt):
        return optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    """Defines and returns function that initializes NN weights
    """
    @abstractmethod
    def weights_init_func(self):
        raise NotImplementedError

    """Defines and a dataset to train on
    """
    @abstractmethod
    def create_dataset(self):
        raise NotImplementedError

    """Defines and a dataloader
    """
    def create_data_loader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.opt.batch_size,
                                           num_workers=self.opt.workers)

    """Saves a generated sample at a specified path
       Parameters:
          path - where to save a sample
    """
    @abstractmethod
    def save_gen_sample(self, path):
        raise NotImplementedError

    """Runs evaluation metrics specific to GAN
       Parameters:
           fake_sample - a sample for generator on fixed noise
           real_sample - a fixed real sample
    """
    @abstractmethod
    def evaluate(self, fake_sample, real_sample):
        raise NotImplementedError

    """Saves statistics specific to GAN
       Parameters:
            fake_sample - a sample for generator on fixed noise
            real_sample - a fixed real sample
    """
    def save_statistics(self, fake_sample, results_folder):
        # will fail for image GAN
        save_kde(fake_sample, self.dataset.distribution, results_folder)

        # Save fake sample log likelihood in case we want to use it later
        with open("{}rs_ll.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.data_log_likelihoods))

        # Save fake sample log likelihood plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.data_log_likelihoods)
        plt.xlabel("Epoch")
        plt.ylabel("Data log likelohood")
        plt.savefig("{}rs_ll.png".format(results_folder))

        # Save fake sample log likelihood in case we want to use it later
        with open("{}hq_rate.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.hq_percentage))

        # Save high quality rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.hq_percentage)
        plt.axhline(y=1, color='tab:red')
        plt.xlabel("Epoch")
        plt.ylabel("High quality rate")
        plt.savefig("{}hq_rate.png".format(results_folder))

        # Save generated sample x stdev in case we want to use it later
        with open("{}x_stdev.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.stdev_x))

        # Save high quality rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.stdev_x)
        plt.axhline(y=self.dataset.distribution.stdev, color='tab:red')
        plt.xlabel("Epoch")
        plt.ylabel("X standard deviation")
        plt.savefig("{}x_stdev.png".format(results_folder))

        # Save generated sample y stdev in case we want to use it later
        with open("{}y_stdev.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.stdev_y))

        # Save high quality rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.stdev_y)
        plt.axhline(y=self.dataset.distribution.stdev, color='tab:red')
        plt.xlabel("Epoch")
        plt.ylabel("Y standard deviation")
        plt.savefig("{}y_stdev.png".format(results_folder))

        # Save generated sample JS-divergence in case we want to use it later
        with open("{}jsd.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.js_divergence))

        # Save high quality rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.js_divergence)
        plt.axhline(y=0, color='tab:red')
        plt.xlabel("Epoch")
        plt.ylabel("JSD(nats)")
        plt.savefig("{}jsd.png".format(results_folder))

    """Returns real sample of the dataset
       Parameters:
            eval_sample_size - size of sample to be provided
            real_sample - a sample of the real data set of size eval_sample_size
    """
    @abstractmethod
    def real_sample(self, eval_sample_size):
        raise NotImplementedError

    """Sample noise of certain size
       Parameters:
            size - size of sample to be provided
            noise_sample - a sample of the noise of size
    """
    @abstractmethod
    def sample_noise(self, size):
        raise NotImplementedError


    """Performs discriminator training step on a batch of real and fake samples
       Parameters:
           model - an instance of GAN class which defines GAN objects
           fake_sample - a batch of generated samples
           real_sample - a batch of real samples
    """
    def train_discriminator(self, fake_sample, real_sample):
        self.d_optimizer.zero_grad()
        real_prediction = self.discriminator(real_sample)
        # get fake sample from generator
        fake_prediction = self.discriminator(fake_sample)

        fake_loss, real_loss = self.d_loss(fake_prediction, real_prediction)
        full_loss = fake_loss + real_loss

        full_loss.backward()
        self.d_optimizer.step()
        return full_loss.item(), real_prediction, fake_prediction

    """Performs generator training step on a batch of real and fake samples
       Parameters:
           model - an instance of GAN class which defines GAN objects
           fake_sample - a batch of generated samples
    """
    def train_generator(self, fake_sample):
        self.g_optimizer.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        d_output = self.discriminator(fake_sample).view(-1)
        # Calculate G's loss based on this output
        g_loss = self.g_loss(d_output)
        # Calculate gradients for G
        g_loss.backward()
        # Update G
        self.g_optimizer.step()
        return g_loss.item(), d_output

    def train(self, results_folder, im_set=False):
        # Create results directory
        try:
            os.mkdir(results_folder)
        except FileExistsError:
            pass

        eval_sample_size = 10000
        fixed_noise = self.sample_noise(eval_sample_size)
        real_sample_fixed = self.real_sample(eval_sample_size)
        num_epochs = self.opt.num_epochs
        print("Starting Training Loop...")
        begin_time = datetime.datetime.now()
        steps_per_epoch = int(np.floor(len(self.data_loader) / self.opt.batch_size))
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            iter = 0
            for i, real_sample in enumerate(self.data_loader, 0):
                if im_set:
                    real_sample = real_sample[0]
                    real_sample = real_sample*2 -1
                ############################
                # (1) Update Discriminator network
                ###########################
                fake_sample = self.generator(self.sample_noise(self.opt.batch_size)).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample)
                self.d_losses.append(d_loss)

                ############################
                # (2) Update Generator network
                ###########################
                fake_sample = self.generator(self.sample_noise(self.opt.batch_size))

                g_loss, fake_out2 = self.train_generator(fake_sample)
                self.g_losses.append(g_loss)
                iter += 1

                # Each few iterations we plot statistics with current discriminator loss, generator loss,
                # mean of discriminator prediction for real sample,
                # mean of discriminator prediction for fake sample before discriminator was trained,
                # mean of discriminator prediction for fake sample after discriminator was trained,
                if iter % 250 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch+1, num_epochs, iter, steps_per_epoch,
                             d_loss, g_loss, real_out.mean().item(), fake_out.mean().item(), fake_out2.mean().item()))

            # After each epoch we save global statistics
            # Sample from generator with fixed noise
            if not im_set:
                with torch.no_grad():
                    fake_fixed = self.generator(fixed_noise)
                fake_shape = fake_fixed.shape
                fake_sample_fixed = fake_fixed.reshape((fake_shape[0], fake_shape[2])).numpy()
            else:
                fake_sample_fixed = self.generator(fixed_noise)

            # Check how the generator is doing by saving its output on fixed_noise
            self.save_gen_sample(fake_sample_fixed, epoch, results_folder)

            # Calculate and save evaluation metrics
            self.evaluate(fake_sample_fixed, real_sample_fixed)

        print(datetime.datetime.now() - begin_time)
        # Losses statistics
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G")
        plt.plot(self.d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("{}train_summary.png".format(results_folder))

        # At the end of training save final stats on fake sample
        fake_fixed = self.generator(fixed_noise).reshape((fake_shape[0], fake_shape[2])).detach().numpy()
        self.save_statistics(fake_fixed, results_folder)


class ToyGAN(GAN):
    def __init__(self, opt):
        super().__init__(opt)
        # Toy data evaluation metrics statistics
        self.data_log_likelihoods = []
        self.hq_percentage = []
        self.stdev_x = []
        self.stdev_y = []
        self.js_divergence = []

    def create_discriminator(self):
        return ToyDiscriminator()

    def create_generator(self):
        return ToyGenerator()

    def weights_init_func(self):
        return weights_init_toy

    def create_dataset(self):
        return MixtureOfGaussiansDataset(SimulatedDistribution(self.opt.toy_type),
                                         self.opt.toy_std,
                                         self.opt.toy_scale,
                                         self.opt.toy_len)

    def save_gen_sample(self, sample, epoch, out_dir):
        path = "{}epoch {}.png".format(out_dir, epoch + 1)
        x, y = extract_xy(sample)
        modes_x, modes_y = extract_xy(self.dataset.distribution.centers())

        plt.figure()
        plt.scatter(x, y, s=1.5)
        plt.scatter(modes_x, modes_y, s=30, marker="D")
        plt.savefig(path)
        plt.close()

    def write_to_writer(self, sample, title, writer, epoch):
        x, y = extract_xy(sample)
        fig = plt.figure()
        fig.scatter(x, y, s=1.5)
        writer.add_figure(title, fig, global_step=epoch)

    def evaluate(self, fake_sample, real_sample):
        # Obtain fixed real data log likelihood estimate based on KDE from fake sample
        real_data_ll = data_log_likelihood(fake_sample, real_sample, self.dataset.distribution.stdev)
        self.data_log_likelihoods.append(real_data_ll)

        # Obtain sample quality statistics
        hq_percentage, stdev, js_diver = self.dataset.distribution.measure_sample_quality(fake_sample)
        self.hq_percentage.append(hq_percentage)
        self.stdev_x.append(stdev[0])
        self.stdev_y.append(stdev[1])
        self.js_divergence.append(js_diver)
    
    def real_sample(self, eval_sample_size):
        return self.dataset.distribution.sample(eval_sample_size)
    
    def sample_noise(self, size):
        noise = -1 * torch.rand(size, 2) + 0.5
        return noise

class CelebWGAN(GAN):
    def __init__(self, opt):
        super().__init__(opt)

    def weights_init_func(self):
        return weights_init

    def create_dataset(self):
        return celeb_dataset(self.opt)

    def save_gen_sample(self, gen_imgs, epoch, out_dir):
        save_image(gen_imgs.data[:25], str(out_dir)+"/epoch%d.png" % (epoch+1), nrow=5, normalize=True)

    def create_d_optimizer(self,opt):
        return optim.RMSprop(self.discriminator.parameters(), lr=self.opt.lr)

    def create_g_optimizer(self,opt):
        return optim.RMSprop(self.generator.parameters(), lr=self.opt.lr)
    
    def create_discriminator(self):
        discriminator = CelebaDiscriminator(self.opt.nc, self.opt.image_size)
        return discriminator.cuda()
    
    def create_generator(self):
        generator = CelebaGenerator(self.opt.nz, self.opt.nc, self.opt.image_size)
        return generator.cuda()
    
    def train_generator(self, fake_sample):
        self.g_optimizer.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        d_output = self.discriminator(fake_sample).view(-1)
        # Calculate G's loss based on this output
        g_loss = -torch.mean(d_output)
        # Calculate gradients for G
        g_loss.backward()
        # Update G
        self.g_optimizer.step()
        return g_loss.item(), d_output
    
    def train_discriminator(self, fake_sample, real_sample):
        self.d_optimizer.zero_grad()
        real_prediction = self.discriminator(real_sample)
        # get fake sample from generator
        fake_prediction = self.discriminator(fake_sample)

        full_loss = -(torch.mean(real_prediction) - torch.mean(fake_prediction))

        full_loss.backward()
        self.d_optimizer.step()
        return full_loss.item(), real_prediction, fake_prediction
    
    def train(self, results_folder, writer=None):
        # Create results directory
        try:
            os.mkdir(results_folder)
        except FileExistsError:
            pass

        
        Tensor = torch.cuda.FloatTensor #torch.FloatTensor
        num_epochs = self.opt.num_epochs
        clip_value = 0.01
        n_critic = 5
        print("Starting Training Loop...")
        steps_per_epoch = int(np.floor(len(self.data_loader) / self.opt.batch_size))
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, (imgs, _) in enumerate(self.data_loader):
                real_sample = Variable(imgs.type(Tensor))
                ############################
                # (1) Update Discriminator network
                ###########################

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.opt.nz))))

                # Generate a batch of images
                fake_sample = self.generator(z).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample)
                self.d_losses.append(d_loss)

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # Train the generator every n_critic iterations
                if i % n_critic == 0:
                    ############################
                    # (2) Update Generator network
                    ###########################
                    fake_sample = self.generator(z)
    
                    g_loss, fake_out2 = self.train_generator(fake_sample)
                    # add loss n_critic times
                    self.g_losses.extend([g_loss]*n_critic)

                # Each few iterations we plot statistics with current discriminator loss, generator loss,
                # mean of discriminator prediction for real sample,
                # mean of discriminator prediction for fake sample before discriminator was trained,
                # mean of discriminator prediction for fake sample after discriminator was trained,
                if i % 250 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch+1, num_epochs, i, steps_per_epoch,
                             d_loss, g_loss, real_out.mean().item(), fake_out.mean().item(), fake_out2.mean().item()))

            # Check how the generator is doing by saving its output
            self.save_gen_sample(fake_sample, epoch, results_folder)

        # Losses statistics
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G")
        plt.plot(self.d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("{}train_summary.png".format(results_folder))

class ToyWGAN(GAN):
    def __init__(self, opt):
        super().__init__(opt)
        # Toy data evaluation metrics statistics
        self.data_log_likelihoods = []
        self.hq_percentage = []
        self.stdev_x = []
        self.stdev_y = []
        self.js_divergence = []

    def weights_init_func(self):
        return weights_init_toy

    def create_dataset(self):
        return MixtureOfGaussiansDataset(SimulatedDistribution(self.opt.toy_type),
                                         self.opt.toy_std,
                                         self.opt.toy_scale,
                                         self.opt.toy_len)

    def save_gen_sample(self, sample, epoch, out_dir):
        path = "{}epoch {}.png".format(out_dir, epoch + 1)
        x, y = extract_xy(sample)
        modes_x, modes_y = extract_xy(self.dataset.distribution.centers())

        plt.figure()
        plt.scatter(x, y, s=1.5)
        plt.scatter(modes_x, modes_y, s=30, marker="D")
        plt.savefig(path)
        plt.close()

    def write_to_writer(self, sample, title, writer, epoch):
        x, y = extract_xy(sample)
        fig = plt.figure()
        fig.scatter(x, y, s=1.5)
        writer.add_figure(title, fig, global_step=epoch)

    def evaluate(self, fake_sample, real_sample):
        # Obtain fixed real data log likelihood estimate based on KDE from fake sample
        real_data_ll = data_log_likelihood(fake_sample, real_sample, self.dataset.distribution.stdev)
        self.data_log_likelihoods.append(real_data_ll)

        # Obtain sample quality statistics
        hq_percentage, stdev, js_diver = self.dataset.distribution.measure_sample_quality(fake_sample)
        self.hq_percentage.append(hq_percentage)
        self.stdev_x.append(stdev[0])
        self.stdev_y.append(stdev[1])
        self.js_divergence.append(js_diver)

    def create_d_optimizer(self,opt):
        return optim.RMSprop(self.discriminator.parameters(), lr=self.opt.lr)

    def create_g_optimizer(self,opt):
        return optim.RMSprop(self.generator.parameters(), lr=self.opt.lr)
    
    def create_discriminator(self):
        return WassersteinToyDiscriminator()
    
    def create_generator(self):
        return WassersteinToyGenerator()
    
    def train_generator(self, fake_sample):
        self.g_optimizer.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        d_output = self.discriminator(fake_sample).view(-1)
        # Calculate G's loss based on this output
        g_loss = -torch.mean(d_output)
        # Calculate gradients for G
        g_loss.backward()
        # Update G
        self.g_optimizer.step()
        return g_loss.item(), d_output
    
    def train_discriminator(self, fake_sample, real_sample):
        self.d_optimizer.zero_grad()
        real_prediction = self.discriminator(real_sample)
        # get fake sample from generator
        fake_prediction = self.discriminator(fake_sample)

        full_loss = -(torch.mean(real_prediction) - torch.mean(fake_prediction))

        full_loss.backward()
        self.d_optimizer.step()
        return full_loss.item(), real_prediction, fake_prediction
    
    def train(self, results_folder, writer=None):
        # Create results directory
        try:
            os.mkdir(results_folder)
        except FileExistsError:
            pass

        eval_sample_size = 10000
        fixed_noise = sample_noise(eval_sample_size)
        real_sample_fixed = self.dataset.distribution.sample(eval_sample_size)
        num_epochs = self.opt.num_epochs
        clip_value = 0.01
        n_critic = 5
        print("Starting Training Loop...")
        steps_per_epoch = int(np.floor(len(self.data_loader) / self.opt.batch_size))
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, real_sample in enumerate(self.data_loader, 0):
                ############################
                # (1) Update Discriminator network
                ###########################
                fake_sample = self.generator(sample_noise(self.opt.batch_size)).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample)
                self.d_losses.append(d_loss)

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                # Train the generator every n_critic iterations
                if (i + 1) % n_critic == 0:
                    ############################
                    # (2) Update Generator network
                    ###########################
                    fake_sample = self.generator(sample_noise(self.opt.batch_size))
    
                    g_loss, fake_out2 = self.train_generator(fake_sample)
                    # add loss n_critic times
                    self.g_losses.extend([g_loss]*n_critic)

                # Each few iterations we plot statistics with current discriminator loss, generator loss,
                # mean of discriminator prediction for real sample,
                # mean of discriminator prediction for fake sample before discriminator was trained,
                # mean of discriminator prediction for fake sample after discriminator was trained,
                if (i + 1) % 250 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch+1, num_epochs, i + 1, steps_per_epoch,
                             d_loss, g_loss, real_out.mean().item(), fake_out.mean().item(), fake_out2.mean().item()))

            # After each epoch we save global statistics
            # Sample from generator with fixed noise
            with torch.no_grad():
                fake_fixed = self.generator(fixed_noise)
            fake_shape = fake_fixed.shape
            fake_sample_fixed = fake_fixed.reshape((fake_shape[0], fake_shape[2])).numpy()

            # Check how the generator is doing by saving its output on fixed_noise
            self.save_gen_sample(fake_sample_fixed, epoch, results_folder)

            # Calculate and save evaluation metrics
            self.evaluate(fake_sample_fixed, real_sample_fixed)

        # Losses statistics
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G")
        plt.plot(self.d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("{}train_summary.png".format(results_folder))

        # At the end of training save final stats on fake sample
        fake_fixed = self.generator(fixed_noise).reshape((fake_shape[0], fake_shape[2])).detach().numpy()
        self.save_statistics(fake_fixed, results_folder)


class ImgGAN(GAN): 
    """
    The ImgGAN class is an abstraction of methods that are different from 
    the toy data set GAN, but similar for all image GANs. This class was 
    introduced to reduce code duplication.
    It might be a good idea to move the dataloader here as well, though 
    it might also be nice to have that below the creation of the data set 
    for code readability.
    """
    
    def save_gen_sample(self, sample, epoch, out_dir):
        fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), sample):
            img = img.detach()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        path = "{}epoch {}.png".format(out_dir, epoch + 1)
        fig.savefig(path)
        plt.close()
    
    def weights_init_func(self):
        return weights_init

class MNISTGAN(ImgGAN):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []
    
    def create_discriminator(self):
        return MNISTDiscriminator(self.opt.input_size, self.opt.d_hidden_size, self.opt.d_output_size)

    def create_generator(self):
        return MNISTGenerator(self.opt.nz, self.opt.g_hidden_size, self.opt.g_output_size)

    def create_dataset(self):
        return mnist_dataset(self.opt)

    def save_gen_sample(self, sample, epoch, out_dir):
        path = "{}epoch {}.png".format(out_dir, epoch + 1)
        x, y = extract_xy(sample)
        modes_x, modes_y = extract_xy(self.dataset.distribution.centers())

    def real_sample(self, eval_sample_size):
        pass

    def sample_noise(self, size):  #size is nz here
        z = np.random.uniform(-1, 1, size=(size, self.opt.nz))
        return torch.from_numpy(z).float()

    def save_statistics(self, fake_sample):
        pass

class CelebGAN(ImgGAN):
    def __init__(self, opt):
        super().__init__(opt)

    def create_dataset(self):
        return celeb_dataset(self.opt)

    def save_gen_sample(self, gen_imgs, epoch, out_dir):
        save_image(gen_imgs.data[:25], str(out_dir)+"/epoch%d.png" % (epoch+1), nrow=5, normalize=True)
    
    def create_discriminator(self):
        discriminator = CelebaDiscriminator(self.opt.nc, self.opt.image_size)
        return discriminator.cuda()
    
    def create_generator(self):
        generator = CelebaGenerator(self.opt.nz, self.opt.nc, self.opt.image_size)
        return generator.cuda()


def main():
    #MNIST
    results_folder = "MNIST gan2/"
    # Change the default parameters if needed
    opt = MNISTGANOptions()
    # Set up your model here
    gan = MNISTGAN(opt)
    print(gan.generator)
    print(gan.discriminator)
    gan.train(results_folder, True)

if __name__ == '__main__':
    main()
