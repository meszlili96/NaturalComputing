import copy
import torch.optim as optim
from data import *
from utils import *
from gen_losses import *
from simdata import ToyGenerator, ToyDiscriminator, weighs_init_toy, save_sample, EightInCircle, Grid, StandardGaussian
from discr_loss import DiscriminatorLoss
from fitness_function import egan_fitness


class EGANOptions():
    def __init__(self, ngpu=0):
        self.num_epochs = 50
        self.ngpu = 0
        self.lr = 1e-03
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.batch_size = 100
        self.workers = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")


class ToyEGANOptions(EGANOptions):
    def __init__(self, ngpu=0):
        super().__init__(ngpu=ngpu)
        self.toy_type = 1
        self.toy_std = 0.2
        self.toy_scale = 2.0
        self.toy_len = 100*self.batch_size


class EGAN():
    __metaclass__ = ABCMeta
    def __init__(self, opt):
        self.opt = opt
        # TODO: how do we select gamma? Fitness quality score lies in [0,1] interval,
        #  diversity score values have higher range
        self.gamma = 0.5
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

        self.g_sample_loglike = []

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

    """Saves a generated sample at a specified path
       Parameters:
          path - where to save a sample
    """
    @abstractmethod
    def save_gen_sample_func(self, path):
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

    def train(self, results_folder):
        fixed_noise = sample_noise(10000)
        fixed_noise_ll = sample_noise(500)
        num_epochs = self.opt.num_epochs
        print("Starting Training Loop...")
        steps_per_epoch = int(np.floor(len(self.dataset) / self.opt.batch_size))
        for epoch in range(num_epochs):
            iter = 0
            # For each batch in the dataloader
            for i, real_sample in enumerate(self.dataset, 0):
                ############################
                # (1) Update Discriminator network
                ###########################
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                fake_sample = self.generator(sample_noise(self.opt.batch_size)).detach()
                d_loss, real_out, fake_out = self.train_discriminator(fake_sample,
                                                                      real_sample.float())
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

                # Evolutionary part. Enumerate through mutations (loss functions)
                for _, loss in enumerate(self.g_losses_list):
                    # Copy the parameters of candidate generator and generator's optimiser
                    self.generator.load_state_dict(self.g_previous)
                    self.g_optimizer.load_state_dict(self.opt_g_previous)
                    # Generate fake samples
                    fake_sample = self.generator(sample_noise(self.opt.batch_size))
                    # Train the current generator
                    g_loss, fake_out2 = self.train_generator(loss, fake_sample)
                    cand_losses_list.append(g_loss)
                    fake_out_2_list.append(fake_out2.mean().item())

                    # Compute fitness score on a sample after training
                    with torch.no_grad():
                        fake_sample_trained = self.generator(sample_noise(self.opt.batch_size))

                    f_q, f_d = egan_fitness(self.discriminator, self.d_loss, fake_sample_trained, real_sample.float())
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

                # save fake sample log likelihood
                with torch.no_grad():
                    fake_fixed_ll = self.generator(fixed_noise_ll)
                fake_shape_ll = fake_fixed_ll.shape
                fake_sample_fixed_ll = fake_fixed_ll.reshape((fake_shape_ll[0], fake_shape_ll[2])).numpy()

                self.g_sample_loglike.append(self.target_distr().likelihood_of(fake_sample_fixed_ll))

                iter += 1

                # Each few iterations we plot statistics with current discriminator loss, generator loss,
                # mean of discriminator prediction for real sample,
                # mean of discriminator prediction for fake sample before discriminator was trained,
                # mean of discriminator prediction for fake sample after discriminator was trained,
                if iter % 100 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLast Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f/D(G(z)): %.4f'
                          % (epoch+1, num_epochs, iter, steps_per_epoch,
                             d_loss, self.g_losses[-1], real_out.mean().item(), fake_out.mean().item(), fake_out_2_mean))

            # Check how the generator is doing by saving G's output on fixed_noise
            # I moved it to the end of epoch, but it can be done based on iter value too
            with torch.no_grad():
                fake_fixed = self.generator(fixed_noise)
            fake_shape = fake_fixed.shape
            self.save_gen_sample(fake_fixed.reshape((fake_shape[0], fake_shape[2])).numpy(),
                                 "{}epoch {}.png".format(results_folder, epoch + 1))

            print("Sample log likelihood {}".format(self.target_distr().likelihood_of(fake_sample_fixed_ll)))
            # gan.write_to_writer(fake.reshape((fake_shape[0], fake_shape[2])).numpy(),
            #                    "epoch {}".format(epoch+1), writer, epoch)

        # Save generator's sample KDE at the end of training
        fake = self.generator(fixed_noise).reshape((fake_shape[0], fake_shape[2])).detach().numpy()
        # TODO: will fail for image GAN
        save_kde(fake, self.target_distr(), results_folder)

        # Save fake sample log likelihood in case we want to use it later
        with open("{}fs_ll.npy".format(results_folder), 'wb') as f:
            np.save(f, np.array(self.g_sample_loglike))

        # Save fake sample log likelihood plot
        plt.figure(figsize=(10, 5))
        plt.title("Fake sample log likelihood")
        plt.plot(self.g_sample_loglike)
        plt.xlabel("iterations")
        plt.ylabel("Log likelohood")
        plt.savefig("{}Fake log likelihood.png".format(results_folder))

        # Train statistics:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.g_losses, label="G")
        plt.plot(self.d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("{}train_summary.png".format(results_folder))

        # Loss functions statistics:
        selected_loss_stat(self.selected_g_loss, results_folder)


class ToyEGAN(EGAN):
    def create_discriminator(self):
        return ToyDiscriminator()

    def create_generator(self):
        return ToyGenerator()

    def weights_init_func(self):
        return weighs_init_toy

    def create_dataset(self):
        return toy_dataset(self.opt)

    def save_gen_sample(self, sample, path):
        save_sample(sample, path)

    # For KDE plot. A better way to do it might exist but I haven't found one
    def target_distr(self):
        distribution = SimulatedDistribution(self.opt.toy_type)
        if distribution == SimulatedDistribution.eight_gaussians:
            return EightInCircle(stdev=self.opt.toy_std, scale=self.opt.toy_scale)
        elif distribution == SimulatedDistribution.twenty_five_gaussians:
            return Grid(stdev=self.opt.toy_std, scale=self.opt.toy_scale)
        elif distribution == SimulatedDistribution.standard_gaussian:
            return StandardGaussian(stdev=self.opt.toy_std, scale=self.opt.toy_scale)
        else:
            raise ValueError


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
    results_folder = "results/"
    # Change the default parameters if needed
    opt = ToyEGANOptions()
    # Set up your model here
    gan = ToyEGAN(opt)
    gan.train(results_folder)


if __name__ == '__main__':
    main()