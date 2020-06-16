import torch.nn.parallel
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from gan import ToyGAN, CelebGAN, ToyOptions, CelebOptions

from torch.utils.tensorboard import SummaryWriter

"""To define a new GAN create generator and discriminator NNs with architecture you need and define
   a subclass of GAN object in gan.py file to return these NNs. Implement other methods appropriately.
   Then, create a shell script with GAN parameters. Change the model to use in main function.
"""

"""Samples noise from unifrom distribution for Generator
"""
def sample_noise(size):
    noise = -1 * torch.rand(size, 2) + 0.5
    return noise


"""Performs discriminator training step on a batch of real and fake samples
   Parameters:
       model - an instance of GAN class which defines GAN objects
       fake_sample - a batch of generated samples
       real_sample - a batch of real samples
"""
def train_discriminator(gan, fake_sample, real_sample):
    gan.d_optimizer.zero_grad()
    real_prediction = gan.discriminator(real_sample)
    # get fake sample from generator
    fake_prediction = gan.discriminator(fake_sample)

    fake_loss, real_loss = gan.d_loss(fake_prediction, real_prediction)
    full_loss = fake_loss + real_loss

    full_loss.backward()
    gan.d_optimizer.step()
    return full_loss.item(), real_prediction, fake_prediction


"""Performs generator training step on a batch of real and fake samples
   Parameters:
       model - an instance of GAN class which defines GAN objects
       fake_sample - a batch of generated samples
"""
def train_generator(gan, fake_sample):
    gan.g_optimizer.zero_grad()
    # Since we just updated D, perform another forward pass of all-fake batch through D
    d_output = gan.discriminator(fake_sample).view(-1)
    # Calculate G's loss based on this output
    g_loss = gan.g_loss(d_output)
    # Calculate gradients for G
    g_loss.backward()
    # Update G
    gan.g_optimizer.step()
    return g_loss.item(), d_output


"""Runs full GAN training
   Parameters:
       gan - an instance of GAN class which defines GAN objects
       opt - an objects which confroms to options duck type:
                num_epochs - number of training epochs to run
                ngpu = 0
                lr - learning rate for training
                beta1 - hyperparameter for Adam optimizers
                beta2 - hyperparameter for Adam optimizers
                g_loss - generator loss function type, 1 - MinMax, 2- heuristic, 3 - least squares
                batch_size - the batch size used in training
                workers - the number of worker threads for loading the data with the DataLoader
                device - torch.device object
            and other GAN specific properties
"""
def train_gan(gan, opt, writer):
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = sample_noise(10000)
    num_epochs = opt.num_epochs
    results_folder = "../results/"
    print("Starting Training Loop...")
    steps_per_epoch = int(np.floor(len(gan.dataset) / opt.batch_size))
    iter = 0
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, real_sample in enumerate(gan.dataset, 0):
            ############################
            # (1) Update Discriminator network
            ###########################
            fake_sample = gan.generator(sample_noise(opt.batch_size)).detach()
            d_loss, real_out, fake_out = train_discriminator(gan,
                                                             fake_sample,
                                                             real_sample.float())
            gan.d_losses.append(d_loss)

            ############################
            # (2) Update Generator network
            ###########################
            fake_sample = gan.generator(sample_noise(opt.batch_size))

            g_loss, fake_out2 = train_generator(gan,
                                                fake_sample)
            gan.g_losses.append(g_loss)
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
        # I moved it to the end of epoch, but it can be done based on iter value too
        with torch.no_grad():
            fake = gan.generator(fixed_noise)
        fake_shape = fake.shape
        gan.save_gen_sample(fake.reshape((fake_shape[0], fake_shape[2])).numpy(),
                            "{}epoch {}.png".format(results_folder, epoch + 1))
        #gan.write_to_writer(fake.reshape((fake_shape[0], fake_shape[2])).numpy(),
        #                    "epoch {}".format(epoch+1), writer, epoch)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gan.g_losses, label="G")
    plt.plot(gan.d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("{}train_summary.png".format(results_folder))


def main():
    # Change the default parameters if needed
    opt = ToyOptions()
    # Set up your model here
    gan = ToyGAN(opt)
    # Set up write for tracking training progress
    writer = SummaryWriter('runs/test')

    train_gan(gan, opt, writer)

if __name__ == '__main__':
    main()
