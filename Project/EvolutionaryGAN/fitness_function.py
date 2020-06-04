import numpy as np
import torch
import torch.nn as nn

# discriminator - discriminator network
# loss - discriminator loss function
# fake_samples - generated samples
# real_samples - reals samples
def egan_fitness(discriminator, loss, fake_samples, real_samples, device="cpu"):
    # we need to compute gradient for diversity fitness score
    # is other way to do it possible?
    for param in discriminator.parameters():
        param.requires_grad = True

    fake_output = discriminator(fake_samples)
    real_output = discriminator(real_samples)

    # fitness quality part is just an expectation of discrimination output on fake samples
    f_q = fake_output.data.mean().cpu().numpy()

    # fitness diversity computes log of discriminator gradient of
    # expectation of logarithm of prediction on real and fake predictions
    # TODO: understand and describe it better
    eval_D_fake, eval_D_real = loss(fake_output, real_output)
    eval_D = eval_D_fake + eval_D_real

    gradients = torch.autograd.grad(outputs=eval_D, inputs=discriminator.parameters(),
                                    grad_outputs=torch.ones(eval_D.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    with torch.no_grad():
        for i, grad in enumerate(gradients):
            grad = grad.view(-1)
            allgrad = grad if i == 0 else torch.cat([allgrad, grad])

    f_d = torch.log(torch.norm(allgrad)).data.cpu().numpy()
    return f_q, f_d