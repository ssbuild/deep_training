# # -*- coding: utf-8 -*-
# # @Time    : 2023/4/27 9:46
# """
# DCGAN - Accelerated with Lightning Fabric
#
# """
# import os
# import time
# from pathlib import Path
#
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.optim as optim
# import torch.utils.data
# import torchvision.utils
# from lightning.fabric import Fabric
# from .engine import PPOEngine
#
#
#
# class PPOTrainer:
#
#     def __init__(self,*args,**kwargs):
#         fabric = Fabric(*args,**kwargs)
#         fabric.launch()
#         output_dir = Path("outputs-fabric", time.strftime("%Y%m%d-%H%M%S"))
#         output_dir.mkdir(parents=True, exist_ok=True)
#
#
#     def fit(self,train_dataloaders=None, val_dataloaders=None, datamodule=None, ckpt_path=None):
#
# def main():
#
#
#
#
#     # Create the generator
#     generator = Generator()
#
#     # Apply the weights_init function to randomly initialize all weights
#     generator.apply(weights_init)
#
#     # Create the Discriminator
#     discriminator = Discriminator()
#
#     # Apply the weights_init function to randomly initialize all weights
#     discriminator.apply(weights_init)
#
#     # Initialize BCELoss function
#     criterion = nn.BCELoss()
#
#     # Create batch of latent vectors that we will use to visualize
#     #  the progression of the generator
#     fixed_noise = torch.randn(64, nz, 1, 1, device=fabric.device)
#
#     # Establish convention for real and fake labels during training
#     real_label = 1.0
#     fake_label = 0.0
#
#     # Set up Adam optimizers for both G and D
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
#     optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
#
#     discriminator, optimizer_d = fabric.setup(discriminator, optimizer_d)
#     generator, optimizer_g = fabric.setup(generator, optimizer_g)
#     dataloader = fabric.setup_dataloaders(dataloader)
#
#     # Lists to keep track of progress
#     losses_g = []
#     losses_d = []
#     iteration = 0
#
#     # Training loop
#     for epoch in range(num_epochs):
#         for i, data in enumerate(dataloader, 0):
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             # (a) Train with all-real batch
#             discriminator.zero_grad()
#             real = data[0]
#             b_size = real.size(0)
#             label = torch.full((b_size,), real_label, dtype=torch.float, device=fabric.device)
#             # Forward pass real batch through D
#             output = discriminator(real).view(-1)
#             # Calculate loss on all-real batch
#             err_d_real = criterion(output, label)
#             # Calculate gradients for D in backward pass
#             fabric.backward(err_d_real)
#             d_x = output.mean().item()
#
#             # (b) Train with all-fake batch
#             # Generate batch of latent vectors
#             noise = torch.randn(b_size, nz, 1, 1, device=fabric.device)
#             # Generate fake image batch with G
#             fake = generator(noise)
#             label.fill_(fake_label)
#             # Classify all fake batch with D
#             output = discriminator(fake.detach()).view(-1)
#             # Calculate D's loss on the all-fake batch
#             err_d_fake = criterion(output, label)
#             # Calculate the gradients for this batch, accumulated (summed) with previous gradients
#             fabric.backward(err_d_fake)
#             d_g_z1 = output.mean().item()
#             # Compute error of D as sum over the fake and the real batches
#             err_d = err_d_real + err_d_fake
#             # Update D
#             optimizer_d.step()
#
#             # (2) Update G network: maximize log(D(G(z)))
#             generator.zero_grad()
#             label.fill_(real_label)  # fake labels are real for generator cost
#             # Since we just updated D, perform another forward pass of all-fake batch through D
#             output = discriminator(fake).view(-1)
#             # Calculate G's loss based on this output
#             err_g = criterion(output, label)
#             # Calculate gradients for G
#             fabric.backward(err_g)
#             d_g_z2 = output.mean().item()
#             # Update G
#             optimizer_g.step()
#
#             # Output training stats
#             if i % 50 == 0:
#                 fabric.print(
#                     f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\t"
#                     f"Loss_D: {err_d.item():.4f}\t"
#                     f"Loss_G: {err_g.item():.4f}\t"
#                     f"D(x): {d_x:.4f}\t"
#                     f"D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}"
#                 )
#
#             # Save Losses for plotting later
#             losses_g.append(err_g.item())
#             losses_d.append(err_d.item())
#
#             # Check how the generator is doing by saving G's output on fixed_noise
#             if (iteration % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
#                 with torch.no_grad():
#                     fake = generator(fixed_noise).detach().cpu()
#
#                 if fabric.is_global_zero:
#                     torchvision.utils.save_image(
#                         fake,
#                         output_dir / f"fake-{iteration:04d}.png",
#                         padding=2,
#                         normalize=True,
#                     )
#                 fabric.barrier()
#
#             iteration += 1
#
#
#
#
#
# if __name__ == "__main__":
#     main()
