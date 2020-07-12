import copy
import torch.optim as optim
from nets import *
from data import *
from utils import *
from gen_losses import *
from simdata import extract_xy
from discr_loss import DiscriminatorLoss
from fitness_function import egan_fitness
from torchvision import datasets
from torchvision.utils import save_image
import datetime

class EGANOptions():
    def __init__(self, ngpu=0):
        self.num_epochs = 100
        self.ngpu = 1
        self.lr = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.batch_size = 64
        self.workers = 1
        self.gamma = 0.05
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")


class ToyEGANOptions(EGANOptions):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.toy_type = 1
        self.toy_std = 0.05
        self.toy_scale = 2.0
        self.toy_len = 500*self.batch_size

class MNISTEGANOptions(EGANOptions):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.dataroot = "data/MNIST"
        self.input_size = 784
        self.d_output_size = 1
        self.d_hidden_size = 32
        self.nz = 100
        self.g_output_size = 784
        self.g_hidden_size = 32


class CelebaEGANOptions(EGANOptions):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.nc = 3
        self.nz = 100
        self.image_size = 64
        self.dataroot = "data/celeba"

class EGAN():
    __metaclass__ = ABCMeta
    def __init__(self, opt):
        self.opt = opt

        self.gamma = opt.gamma
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
        # Types of selected generator losses for each training step
        self.selected_g_loss = []

        # Initialize Discriminator loss functions
        self.d_loss = self.create_d_loss()
        # Initialize Generator loss functions (mutations)
        self.g_losses_list = [Minmax(), Heuristic(), LeastSquares()]

        # Setup Adam optimizers for both G and D
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

        # To keep the best individual parameters during evolution
        self.g_previous = copy.deepcopy(self.generator.state_dict())
        self.opt_g_previous = copy.deepcopy(self.g_optimizer.state_dict())

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

    def train_generator(self, loss, fake_sample):
        self.g_optimizer.zero_grad()

        for param in self.discriminator.parameters():
            param.requires_grad = False

        # Since we just updated D, perform another forward pass of all-fake batch through D
        d_output = self.discriminator(fake_sample).view(-1)
        # Calculate G's loss based on this output
        g_loss = loss(d_output)
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
        fitness_sample_size = 1024
        fixed_noise = self.sample_noise(eval_sample_size)
        real_sample_fixed = self.real_sample(eval_sample_size)
        print(self.gamma)
        ## for testing save function
        fake_sample_fixed = self.generator(fixed_noise)
        if not im_set:
            fake_shape = fake_sample_fixed.shape
            fake_sample_fixed = fake_sample_fixed.reshape((fake_shape[0], fake_shape[2])).detach().numpy()
        self.save_gen_sample(fake_sample_fixed, -1, results_folder)
        ## end of testing save function
        
        num_epochs = self.opt.num_epochs
        print("Starting Training Loop...")
        begin_time = datetime.datetime.now()
        steps_per_epoch = int(np.floor(len(self.data_loader) / self.opt.batch_size))
        for epoch in range(num_epochs):
            iter = 0
            # For each batch in the dataloader
            for i, real_sample in enumerate(self.data_loader, 0):
                ############################
                # If it is an image dataset, the data_loader needs to be handled differently, and transformation is needed
                ###########################
                if im_set:
                    real_sample = real_sample[0]
                    real_sample = real_sample*2 -1
                ############################
                # (1) Update Discriminator network
                ###########################
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                fake_sample = self.generator(self.sample_noise(self.opt.batch_size)).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample)
                self.d_losses.append(d_loss)

                ############################
                # (2) Update Generator network
                ###########################
                # Fitness scores for each mutation
                F_scores = []
                # To save trained generators and their optimizers
                g_list = []
                opt_g_list = []
                # Losses of different candidates
                cand_losses_list = []
                # Mean discriminator output on the samples from candidates generators
                fake_out_2_list = []

                # Generate noise to evaluate on before and after training step, same for each offspring
                noise = self.sample_noise(self.opt.batch_size)
                # Evolutionary part. Enumerate through mutations (loss functions)
                for _, loss in enumerate(self.g_losses_list):
                    # Copy the parameters of candidate generator and generator's optimiser
                    self.generator.load_state_dict(self.g_previous)
                    self.g_optimizer.load_state_dict(self.opt_g_previous)

                    # Train the current generator
                    fake_sample = self.generator(self.sample_noise(self.opt.batch_size))
                    g_loss, fake_out2 = self.train_generator(loss, fake_sample)
                    cand_losses_list.append(g_loss)
                    fake_out_2_list.append(fake_out2.mean().item())

                    # Compute fitness score on a sample after training
                    with torch.no_grad():
                        fake_sample_trained = self.generator(self.sample_noise(fitness_sample_size))

                    f_q, f_d = egan_fitness(self.discriminator, self.d_loss, fake_sample_trained, real_sample)
                    fitness_score = f_q + self.gamma*f_d
                    F_scores.append(fitness_score)

                    # Save an individual
                    g_list.append(copy.deepcopy(self.generator.state_dict()))
                    opt_g_list.append(copy.deepcopy(self.g_optimizer.state_dict()))

                # Select best individual and its optimizer based on fitness score
                best_individual_index = F_scores.index(max(F_scores))
                best_individual = g_list[best_individual_index]
                best_individual_optim = opt_g_list[best_individual_index]
                # Load these states to generator and optimizer
                self.generator.load_state_dict(best_individual)
                self.g_optimizer.load_state_dict(best_individual_optim)
                # Record which mutation was selected for statistics
                self.selected_g_loss.append(best_individual_index + 1)
                # Add loss of the best individual to statistics
                self.g_losses.append(cand_losses_list[best_individual_index])
                # Select the discriminator result on the output of the best individual for statistics
                fake_out_2_mean = fake_out_2_list[best_individual_index]
                # Save best individual for next evolution
                self.g_previous = copy.deepcopy(best_individual)
                self.opt_g_previous = copy.deepcopy(best_individual_optim)

                iter += 1
                # Each few iterations we plot statistics with current discriminator loss, generator loss,
                # mean of discriminator prediction for real sample,
                # mean of discriminator prediction for fake sample before discriminator was trained,
                # mean of discriminator prediction for fake sample after discriminator was trained,
                if iter % 250 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLast Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f/D(G(z)): %.4f'
                          % (epoch+1, num_epochs, iter, steps_per_epoch,
                             d_loss, self.g_losses[-1], real_out.mean().item(), fake_out.mean().item(), fake_out_2_mean))

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

        # Loss functions statistics:
        selected_loss_stat(self.selected_g_loss, results_folder)


class ToyEGAN(EGAN):
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
            np.save(f, np.array(self.stdev_x))

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

class ImgGAN(EGAN): 
    """
    The ImgGAN class is an abstraction of methods that are different from 
    the toy data set GAN, but similar for all image GANs. This class was 
    introduced to reduce code duplication.
    It might be a good idea to move the dataloader here as well, though 
    it might also be nice to have that below the creation of the data set 
    for code readability.
    """
    
    def weights_init_func(self):
        return weights_init

    def create_data_loader(self):
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.opt.batch_size,
                                           shuffle=True,
                                           num_workers=self.opt.workers)
    

class MNISTEGAN(ImgGAN):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []
    
    def create_discriminator(self):
        return MNISTDiscriminator(self.opt.input_size, self.opt.d_hidden_size, self.opt.d_output_size)

    def create_generator(self):
        return MNISTGenerator(self.opt.nz, self.opt.g_hidden_size, self.opt.g_output_size)

    def create_dataset(self):
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transform)
        return train_data

    def evaluate(self, fake_sample, real_sample):
        pass

    def real_sample(self, eval_sample_size):
        pass

    def sample_noise(self, size):  #size is nz here
        z = np.random.uniform(-1, 1, size=(size, self.opt.nz))
        return torch.from_numpy(z).float()

    def save_statistics(self, fake_sample):
        pass

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


class CelebaEGAN(ImgGAN):
    def __init__(self, opt):
        super().__init__(opt)
        self.img_list = []
    
    def create_discriminator(self):
        return CelebaDiscriminator(self.opt.nc, self.opt.image_size)

    def create_generator(self):
        return CelebaGenerator(self.opt.nz, self.opt.nc, self.opt.image_size)

    def create_dataset(self):
        #transform = transforms.ToTensor()
        #train_data = datasets.CelebA(root='data/celeba/celeba', split='train', download=False, transform=transform)
        return image_dataset(self.opt)
        #return train_data

    def evaluate(self, fake_sample, real_sample):
        pass

    def real_sample(self, eval_sample_size):
        pass

    def sample_noise(self, size): 
        z = np.random.uniform(0, 1, size=(size, self.opt.nz))
        return torch.from_numpy(z).float()

    def save_statistics(self, fake_sample):
        pass

    def save_gen_sample(self, sample, epoch, out_dir):
        save_image(sample.data[:25], str(out_dir)+"/epoch%d.png" % (epoch+1), nrow=5, normalize=True)


def selected_loss_stat(selected_g_losses, results_folder):
    selected_g_losses = np.array(selected_g_losses)
    groups_num = 5
    grouped_by_steps = np.split(selected_g_losses, groups_num)
    minmax_counts = [(x == 1).sum() for x in grouped_by_steps]
    heuristic_counts = [(x == 2).sum() for x in grouped_by_steps]
    ls_counts = [(x == 3).sum() for x in grouped_by_steps]

    ind = np.arange(groups_num)  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    _ = ax.bar(ind - width, minmax_counts, width, label='MinMax')
    _ = ax.bar(ind, heuristic_counts, width, label='Heuristic')
    _ = ax.bar(ind + width, ls_counts, width, label='Least Square')

    ax.set_title('Selected mutations')
    ax.set_xticks(ind)
    ax.set_xticklabels(('0-2', '2-4', '4-6', '6-8', '8-10'))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    plt.savefig("{}selected_mutations_stats.png".format(results_folder))
    plt.close()


def main():

    set_seed()
    # 8 gaussians
    results_folder = "8 gauss 0.2 egan/"
    # Change the default parameters if needed
    opt = ToyEGANOptions()
    # Set up your model here
    gan = ToyEGAN(opt)
    gan.train(results_folder)
    
    # 25 gaussians
    results_folder = "25 gauss egan/"
    # Change the default parameters if needed
    opt = ToyEGANOptions()
    opt.toy_type = 2
    # Set up your model here
    gan = ToyEGAN(opt)
    gan.train(results_folder)


    """
    #pokemon
    results_folder = "poke egan/"
    # Change the default parameters if needed
    opt = PokeEGANOptions()
    #opt.toy_type = 2
    # Set up your model here
    gan = PokeEGAN(opt)
    gan.train(results_folder)
    """
    
    
    #MNIST
    results_folder = "MNIST egan3/"
    # Change the default parameters if needed
    opt = MNISTEGANOptions()
    # Set up your model here
    gan = MNISTEGAN(opt)
    print(gan.generator)
    print(gan.discriminator)
    gan.train(results_folder, True)
    
    """
    #Celeba
    results_folder = "Celeba egan/"
    # Change the default parameters if needed
    opt = CelebaEGANOptions()
    # Set up your model here
    gan = CelebaEGAN(opt)
    print(gan.generator)
    print(gan.discriminator)
    gan.train(results_folder, True)
    """

if __name__ == '__main__':
    main()
