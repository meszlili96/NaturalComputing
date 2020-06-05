import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Class to define loss functions for Discriminator
# call returns Discriminator on generated and real samples
class DiscriminatorLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(DiscriminatorLoss, self).__init__()
        # we it is just a way to create properties named real_label and fake_label
        # it is used in the paper autor's implementation, not sure why
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCELoss()

    """Create label tensors with the same size as the input.
            Parameters:
                labels_dim - the dimensionality of prediction from a discriminator
                target_is_real (bool) - if the ground truth label is for real images or fake images
            Returns:
                A label tensor filled with ground truth label, and with the size of the input
        """

    def get_target_labels(self, labels_dim, target_is_real):
        # tensor of size 1
        target_label = self.real_label if target_is_real else self.fake_label
        # create the tensor of len(prediction) with target_label values
        return target_label.repeat(labels_dim)

    def __call__(self, fake_prediction, real_prediction):
        fake_labels = self.get_target_labels(fake_prediction.shape, False)
        reals_labels = self.get_target_labels(real_prediction.shape, True)

        loss_fake = self.loss(fake_prediction, fake_labels)
        loss_real = self.loss(real_prediction, reals_labels)

        return loss_fake, loss_real


def main():
    # Generator loss functions demonstration:
    output_dim = 10
    outputs_fake = np.linspace(0, 1, 100)
    outputs_real = np.linspace(0, 1, 100)
    N = len(outputs_fake)

    fake, real = np.meshgrid(outputs_fake, outputs_real)

    loss = DiscriminatorLoss()
    losses_fake = np.zeros(fake.shape)
    losses_real = np.zeros(fake.shape)
    losses = np.zeros(fake.shape)
    for i in range(0, N):
        for j in range(0, N):
            fake_pred = fake[i, j]
            real_pred = real[i, j]
            fake_tensor = torch.tensor(fake_pred).repeat(output_dim).float()
            real_tensor = torch.tensor(real_pred).repeat(output_dim).float()
            loss_fake, loss_real = loss(fake_tensor, real_tensor)
            losses_fake[i, j] = loss_fake.item()
            losses_real[i, j] = loss_real.item()
            losses[i, j] = (loss_fake + loss_real).item()

    plt.plot(outputs_fake, losses_fake[0, :], label="Fake loss")
    plt.plot(outputs_real, losses_real[:, 0], label="Real loss")
    plt.xlabel('Discriminator output')
    plt.ylabel('Loss')
    # We zoom in by y axis, otherwise least squares loss is invisible
    plt.legend()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(fake, real, losses, rstride=1, cstride=1, cmap='jet')
    ax.set_xlabel("Fake predictions")
    ax.set_ylabel("Real predictions")
    ax.set_zlabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()