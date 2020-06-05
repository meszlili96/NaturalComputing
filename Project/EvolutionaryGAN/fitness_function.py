import numpy as np
import torch
import torch.nn as nn
from simdata import EightInCircle, StandardGaussian
from discr_loss import DiscriminatorLoss

# discriminator - discriminator network
# criterion - discriminator loss function
# fake_samples - generated samples
# real_samples - reals samples
def egan_fitness(discriminator, criterion, fake_samples, real_samples, device="cpu"):
    # we need to compute gradient for diversity fitness score
    # is other way to do it possible?
    for param in discriminator.parameters():
        param.requires_grad = True

    fake_output = discriminator(fake_samples)
    real_output = discriminator(real_samples)

    # fitness quality part is just an expectation of discrimination output on fake samples
    f_q = fake_output.data.mean().cpu().numpy()

    # fitness diversity computes log of discriminator loss function gradient (see overleaf for more details)
    # BCE for fake and real samples separately
    eval_D_fake, eval_D_real = criterion(fake_output, real_output)
    # full BCE
    eval_D = eval_D_fake + eval_D_real

    # gradients computation, see docs https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
    # TODO: rewrite the comment for final version
    # So, according to the documentation:
    # outputs - the result of differentiatiable function for which we want to compute gradienst
    # inputs - tensonrs for which the gradiesnt will be returned, so we compute gradient of output w.r.t. inputs
    # only_inputs - do not compute and accumulate in .grad any other gradients for tesnsors not from inputs list
    # grad_outputs - an external gradient to start gradients calculations. More detailes are here https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95
    # I don't have full understanding yet, but it should almost always be a tensor of ones unless we wan weighted grads
    # So, with this call we just want to get the gradiens of loss function value for all NN weights,
    # that is why we pass all the network parameters as inputs
    gradients = torch.autograd.grad(outputs=eval_D, inputs=discriminator.parameters(),
                                    grad_outputs=torch.ones(eval_D.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)

    # stack gradients into array
    with torch.no_grad():
        for i, grad in enumerate(gradients):
            grad = grad.view(-1)
            allgrad = grad if i == 0 else torch.cat([allgrad, grad])

    # - log norm computation
    f_d = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
    return f_q, f_d


# Handles our toy Gaussian data sets
class DummyDiscriminator(nn.Module):
    def __init__(self):
        super(DummyDiscriminator, self).__init__()
        # Number of input features is 2, since we work with 2d gaussians
        # input layer
        self.layer_1 = nn.Linear(2, 32)
        # hidden layer
        self.layer_2 = nn.Linear(32, 64)
        # hidden layer
        self.layer_3 = nn.Linear(64, 128)
        # output layer
        self.layer_out = nn.Linear(128, 1)
        # activation
        self.relu = nn.ReLU()
        # binary classification activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.sigmoid(self.layer_out(x))
        return x

def train(discriminator, loss, real_distr):
    n_epochs = 20
    batch_per_epoch = 400
    batch_size = 10
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    discriminator.train()
    for epoch in range(0, n_epochs):
        epoch_loss = 0
        for batch in range(0, batch_per_epoch):
            optimizer.zero_grad()

            real_sample = real_distr.sample(batch_size)
            real_prediction = discriminator(torch.tensor(real_sample).float())
            # for simplicity fake samples are from uniform distr
            fake_sample = -2*torch.rand(batch_size, 2) + 1
            fake_prediction = discriminator(torch.tensor(fake_sample).float())

            fake_loss, real_loss = loss(fake_prediction, real_prediction)
            lv = fake_loss + real_loss
            epoch_loss += lv.item()

            lv.backward()
            optimizer.step()
        print("Epoch {}, loss {}".format(epoch + 1, epoch_loss))


def compute_fitness(discriminator, loss, real_samples, gamma=1):
    bad_q_scores = []
    bad_d_scores = []
    perfect_scores = []
    # To preven getting random results average across a few runs
    for _ in range(0, 30):
        # standart gaussian with stdev 0.5 (against 0.2 for mixture), bad quality, better diversity
        generator_simulation = StandardGaussian(stdev=0.5)
        fake_samples = generator_simulation.sample(10)
        f_q, f_d = egan_fitness(discriminator, loss, torch.tensor(fake_samples).float(),
                                torch.tensor(real_samples).float())
        bad_q_scores.append(f_q + gamma * f_d)

        # one gaussian fom 8 gaussians distr, good quality, bad diversity
        generator_simulation = StandardGaussian(center=(1, 0))
        fake_samples = generator_simulation.sample(10)
        f_q, f_d = egan_fitness(discriminator, loss, torch.tensor(fake_samples).float(),
                                torch.tensor(real_samples).float())
        bad_d_scores.append(f_q + gamma * f_d)

        # Then fitness os perfect generator
        generator_simulation = EightInCircle()
        fake_samples = generator_simulation.sample(10)
        f_q, f_d = egan_fitness(discriminator, loss, torch.tensor(fake_samples).float(),
                                torch.tensor(real_samples).float())
        perfect_scores.append(f_q + gamma * f_d)

    bad_q_scores = np.array(bad_q_scores)
    print("Bad quality generator average fitness {}, std {}".format(bad_q_scores.mean(), bad_q_scores.std()))
    bad_d_scores = np.array(bad_d_scores)
    print("Bad diversity generator average fitness {}, std {}".format(bad_d_scores.mean(), bad_d_scores.std()))
    perfect_scores = np.array(perfect_scores)
    print("Perfect generator average fitness {}, std {}".format(perfect_scores.mean(), perfect_scores.std()))

def main():
    # Generator Fitness function demonstration
    # We assume that real distribution is 8 gaussians and our generator learned only one center
    # We first train the discriminator to learn 8 gaussians against unifrom
    real_distr = EightInCircle()
    real_samples = real_distr.sample(10)
    discriminator = DummyDiscriminator()
    loss = DiscriminatorLoss()
    print("Training the discriminator on eight gaussians in circle:")
    train(discriminator, loss, real_distr)

    discriminator.eval()
    print("Quality and diversity have equal weight:")
    compute_fitness(discriminator, loss, real_samples, gamma=1)
    print("Zero diversity weight:")
    compute_fitness(discriminator, loss, real_samples, gamma=0)
    print("Gamma 0.5:")
    compute_fitness(discriminator, loss, real_samples, gamma=0.5)


if __name__ == '__main__':
    main()