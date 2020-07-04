import torch.optim as optim
from nets import *
from data import *
from utils import *
from gen_losses import *
from simdata import ToyGenerator, ToyDiscriminator, weighs_init_toy, extract_xy
from discr_loss import DiscriminatorLoss


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
        self.ndf = 64
        self.ngf = 64
        self.nz = 100
        self.image_size = 64
        self.dataroot = "celeba"


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
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

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
    @abstractmethod
    def create_data_loader(self):
        raise NotImplementedError

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
    @abstractmethod
    def save_statistics(self, fake_sample):
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
        print("Starting Training Loop...")
        steps_per_epoch = int(np.floor(len(self.data_loader) / self.opt.batch_size))
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            iter = 0
            for i, real_sample in enumerate(self.data_loader, 0):
                ############################
                # (1) Update Discriminator network
                ###########################
                fake_sample = self.generator(sample_noise(self.opt.batch_size)).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample)
                self.d_losses.append(d_loss)

                ############################
                # (2) Update Generator network
                ###########################
                fake_sample = self.generator(sample_noise(self.opt.batch_size))

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
        return weighs_init_toy

    def create_dataset(self):
        return MixtureOfGaussiansDataset(SimulatedDistribution(self.opt.toy_type),
                                         self.opt.toy_std,
                                         self.opt.toy_scale,
                                         self.opt.toy_len)

    def create_data_loader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.opt.batch_size,
                                           num_workers=self.opt.workers)

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

    def save_statistics(self, fake_sample, results_folder):
        # will fail for image GAN
        save_kde(fake_sample, self.dataset.distribution, results_folder, "test")

        # Save fake sample log likelihood in case we want to use it later
        with open("{}fs_ll.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.data_log_likelihoods))

        # Save fake sample log likelihood plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.data_log_likelihoods)
        plt.xlabel("Epoch")
        plt.ylabel("Data log likelohood")
        plt.savefig("{}Real log likelihood.png".format(results_folder))

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


class CelebGAN(GAN):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []

    def create_discriminator(self):
        return Discriminator(self.opt.ngpu, self.opt.nc, self.opt.ndf).to(self.opt.device)

    def create_generator(self):
        return Generator(self.opt.ngpu, self.opt.nc, self.opt.nz, self.opt.ngf).to(self.opt.device)

    def weights_init_func(self):
        return weights_init_celeb

    def create_dataset(self):
        return image_dataset(self.opt)

    def create_data_loader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.opt.batch_size,
                                           shuffle=True,
                                           num_workers=self.opt.workers)

    def evaluate(self, fake_sample, real_sample):
        pass

    def save_statistics(self, fake_sample):
        pass

    def save_gen_sample(self, sample, epoch, out_dir):
        plt.figure()
        plt.imshow(sample)
        plt.savefig(path)
        plt.close()
