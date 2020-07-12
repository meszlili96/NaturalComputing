import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import islice

from discr_loss import DiscriminatorLoss
from gen_losses import Minmax
from simdata import MixtureOfGaussiansDataset, SimulatedDistribution
from nets import ToyDiscriminator, ToyGenerator, weights_init_toy
from utils import sample_noise

def train_discriminator(discriminator, loss, optimizer, fake_sample, real_sample):
    optimizer.zero_grad()
    real_prediction = discriminator(real_sample)
    # get fake sample from generator
    fake_prediction = discriminator(fake_sample)

    fake_loss, real_loss = loss(fake_prediction, real_prediction)
    full_loss = fake_loss + real_loss

    full_loss.backward()
    optimizer.step()
    return full_loss.item(), real_prediction, fake_prediction


def train_generator(loss, discriminator, optimizer, fake_sample):
    optimizer.zero_grad()
    # Since we just updated D, perform another forward pass of all-fake batch through D
    d_output = discriminator(fake_sample).view(-1)
    # Calculate G's loss based on this output
    g_loss = loss(d_output)
    # Calculate gradients for G
    g_loss.backward()
    # Update G
    optimizer.step()
    return g_loss.item(), d_output


def main():
    generator = ToyGenerator()
    generator.apply(weights_init_toy)
    generator_loss = Minmax()

    discriminator = ToyDiscriminator()
    discriminator.apply(weights_init_toy)
    discriminator_loss = DiscriminatorLoss()

    batch_size = 100
    iterable_dataset = MixtureOfGaussiansDataset(SimulatedDistribution.eight_gaussians)
    data_loader = DataLoader(iterable_dataset, batch_size=batch_size)

    # Setup Adam optimizers for both G and D
    learning_rate = 1e-3
    beta1 = 0.5
    beta2 = 0.999
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    steps_per_epoch = 1000
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    g_losses = []
    d_losses = []
    print("Starting Training Loop...")
    fixed_noise = sample_noise(10000)
    num_epochs = 30
    data_folder = "results/"
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        iter = 0
        for real_sample in islice(data_loader, steps_per_epoch):
            ############################
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            fake_sample = generator(sample_noise(batch_size)).detach()
            d_loss, real_out, fake_out = train_discriminator(discriminator,
                                                             discriminator_loss,
                                                             d_optimizer,
                                                             fake_sample,
                                                             real_sample.float())
            d_losses.append(d_loss)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            fake_sample = generator(sample_noise(batch_size))

            g_loss, fake_out2 = train_generator(generator_loss,
                                                discriminator,
                                                g_optimizer,
                                                fake_sample)
            g_losses.append(g_loss)
            # Output training stats
            iter += 1

            # Each few iterations we plot statistics with current discriminator loss, generator loss,
            # mean of discriminator prediction for real sample,
            # mean of discriminator prediction for fake sample before discriminator was trained,
            # mean of discriminator prediction for fake sample after discriminator was trained,
            if iter % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, iter, steps_per_epoch,
                d_loss, g_loss, real_out.mean().item(), fake_out.mean().item(), fake_out2.mean().item()))


        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = generator(fixed_noise)
        fake_shape = fake.shape
        save_sample(fake.reshape((fake_shape[0], fake_shape[2])).numpy(), "{}epoch {}.png".format(data_folder, epoch+1))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("{}train_summary.png".format(data_folder))


if __name__ == '__main__':
    main()