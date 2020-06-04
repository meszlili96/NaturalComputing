import torch.nn.parallel
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from data import *
from parser import Parser
from model import Model

if __name__ == '__main__':
    opt = Parser().parse()
    if opt.dataroot is not None:
        dataset = create_dataset(opt)
    else:
        dataset = toy_dataset(SimulatedDistribution.eight_gaussians)

    model = Model(opt)

    iters = 0
    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=opt.device)
    print("Starting Training Loop...")
    for epoch in range(1):
        # For each batch in the dataloader
        for i, data in enumerate(dataset, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # TODO: I'm not sure I did not break it! But we can just roll back with git
            ## Train with all-real batch
            model.netD.zero_grad()
            # Format batch
            # Real data
            real_cpu = data[0].to(opt.device)
            b_size = real_cpu.size(0)
            # TODO: is this variable used?
            label = torch.full((b_size,), model.real_label, device=opt.device)
            # Forward pass real batch through D
            output_real = model.netD(real_cpu).view(-1)
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, opt.nz, 1, 1, device=opt.device)
            # Generate fake image batch with G
            fake = model.netG(noise)
            # TODO: is this variable used?
            label.fill_(model.fake_label)
            # Classify all fake batch with D
            output_fake = model.netD(fake.detach()).view(-1)

            # Calculate loss for fake and real sample
            errD_fake, errD_real = model.criterion(output)
            # Backpropagate full error
            errD = errD_fake + errD_real
            errD.backward()

            D_x = output_real.mean().item()
            D_G_z1 = output.mean().item()

            # Update D
            model.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.netG.zero_grad()
            label.fill_(model.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = model.g_criterion(output)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            model.optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, opt.num_epochs, i, len(dataset),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            model.G_losses.append(errG.item())
            model.D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataset)-1)):
                with torch.no_grad():
                    fake = model.netG(fixed_noise).detach().cpu()
                model.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(model.G_losses,label="G")
    plt.plot(model.D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
