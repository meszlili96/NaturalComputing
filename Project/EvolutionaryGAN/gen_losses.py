import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

# Abstract class to define custom loss functions for Generator
# call returns Generator loss on Discriminator output for fake samples
class GeneratorLoss(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GeneratorLoss, self).__init__()
        self.loss_func = self.init_loss()
        # we it is just a way to create properties named real_label and fake_label
        # it is used in the paper autor's implementation, not sure why
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    # subclasses should return the loss function from this method
    @abstractmethod
    def init_loss(self):
        raise NotImplementedError

    # sunblasses should calculate loss and return it from this method
    @abstractmethod
    def loss(self, fake_prediction):
        raise NotImplementedError

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

    """Calculate loss given Discriminator's output on fake samples
        Parameters:
            fake_prediction (tensor) - the prediction output from a discriminator
        Returns:
            the calculated loss.
    """
    def __call__(self, fake_prediction):
        return self.loss(fake_prediction)


class Minmax(GeneratorLoss):
    def init_loss(self):
        return nn.BCELoss()

    # fake_prediction (tensor) - the prediction output from a discriminator
    def loss(self, fake_prediction):
        # minmax objective function is equivalen to -BCE with target labels of fake samples (0)
        fake_labels = self.get_target_labels(fake_prediction.shape, False)
        loss_fake = -self.loss_func(fake_prediction, fake_labels)
        return loss_fake


class Heuristic(GeneratorLoss):
    def init_loss(self):
        return nn.BCELoss()

    # fake_prediction (tensor) - the prediction output from a discriminator
    def loss(self, fake_prediction):
        # heuristic objective function is equivalen to BCE with target labels of real samples (1)
        real_labels = self.get_target_labels(fake_prediction.shape, True)
        loss_fake = self.loss_func(fake_prediction, real_labels)
        return loss_fake


class LeastSquares(GeneratorLoss):
    def init_loss(self):
        return nn.MSELoss()

    # fake_prediction (tensor) - the prediction output from a discriminator
    def loss(self, fake_prediction):
        # lest square mutation is just a MSE of discriminator output for fake samples and
        # labels of real samples (1)
        real_labels = self.get_target_labels(fake_prediction.shape, True)
        loss_fake = self.loss_func(fake_prediction, real_labels)
        return loss_fake


def main():
    # Generator loss functions demonstration:
    output_dim = 10
    outputs = np.linspace(0, 1, 1000)

    minmax_loss = Minmax()
    heuristic_loss = Heuristic()
    least_squares_loss = LeastSquares()

    minmax_losses = []
    heuristic_losses = []
    least_squares_losses = []
    for output in outputs:
        out_tensor = torch.tensor(output).repeat(output_dim).float()
        minmax_losses.append(minmax_loss(out_tensor).item())
        heuristic_losses.append(heuristic_loss(out_tensor).item())
        least_squares_losses.append(least_squares_loss(out_tensor).item())

    plt.plot(outputs, minmax_losses, label="Minmax")
    plt.plot(outputs, heuristic_losses, label="Heuristic")
    plt.plot(outputs, least_squares_losses, label="Least Squares")
    plt.xlabel('Discriminator output')
    plt.ylabel('Loss')
    # We zoom in by y axis, otherwise least squares loss is invisible
    plt.axis([0, 1, -4, 4])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()