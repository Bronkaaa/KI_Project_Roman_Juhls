import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from NeuralNetworkBase import NeuralNetworkBase
import matplotlib.pyplot as plt


# Number of GPUs available. Use 0 for CPU mode.



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGenerator(nn.Module):
    def __init__(self, z_dim, num_channels, ngf):
        super(DCGenerator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf*4, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, num_channels, 3, 2, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)
    
    
class DCDiscriminator(nn.Module):
    def __init__(self, num_channels, ndf):
        super(DCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 3 x 3
            nn.Conv2d(ndf*4, 1, 4, 2, 1, bias=False),
            #nn.Sigmoid() # not needed with nn.BCEWithLogitsLoss()
        )

    def forward(self, input):
        return self.main(input)
    
    
    
    
class DCGeneratorCIFAR(nn.Module):
    def __init__(self, z_dim, num_channels, ngf):
        super(DCGeneratorCIFAR, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)




class DCDiscriminatorCIFAR(nn.Module):
    def __init__(self, num_channels, ndf):
        super(DCDiscriminatorCIFAR, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
    


class DCGAN(NeuralNetworkBase):
    
    def __init__(self, z_dim, num_channels, ngf, ndf):
        super(DCGAN, self).__init__()

        
        if num_channels == 1:
            self.generator = DCGenerator(z_dim, num_channels, ngf)
            self.discriminator = DCDiscriminator(num_channels, ndf)
        elif num_channels == 3:
            self.generator = DCGeneratorCIFAR(z_dim, num_channels, ngf)
            self.discriminator = DCDiscriminatorCIFAR(num_channels, ndf)
        
        

    def test_model():
        return None
    
    def train_model(self, train_loader, num_channels, num_epochs, z_dim, criterion, optimizerD, optimizerG, saturating=False):
        
        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        fixed_noise = torch.randn(64, z_dim, 1, 1)
        
        
        ## Create the generator
        #netG = DCGenerator(ngpu).to(device)
        netG = self.generator


        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netG.apply(weights_init)
        
        ## Create the Discriminator
        #netD = DCDiscriminator(ngpu).to(device)
        netD = self.discriminator

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)


        
        ## Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        G_grads_mean = []
        G_grads_std = []
        D_losses = []

        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(train_loader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0]
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, z_dim, 1, 1)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                
                if saturating:
                    label.fill_(fake_label) # Saturating loss: Use fake_label y = 0 to get J(G) = log(1−D(G(z)))
                else:
                    label.fill_(real_label) # Non-saturating loss: fake labels are real for generator cost
                
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                
                if saturating:
                    errG = -criterion(output, label) # Saturating loss: -J(D) = J(G)
                else:
                    errG = criterion(output, label) # Non-saturating loss
                
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                
                # Save gradients
            G_grad = [p.grad.view(-1).cpu().numpy() for p in list(netG.parameters())]
            G_grads_mean.append(np.concatenate(G_grad).mean())
            G_grads_std.append(np.concatenate(G_grad).std())


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise



            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            #test
            if num_channels == 1:
                self.plot_generated_images(num_channels, img_list, epoch+1, num_images=16, img_size=(28, 28))
            elif num_channels == 3:
                self.plot_generated_images(num_channels, img_list, epoch+1, num_images=16, img_size=(32, 32))

            
            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch+1), num_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        
        return G_losses, D_losses, G_grads_mean, G_grads_std, img_list
    
    
    def plot_generated_images(self,num_channels, img_list, epoch, num_images=16, img_size=(28, 28)):
        generated_images = img_list[-1]  # Get the latest generated images
        generated_images = generated_images.cpu().numpy().transpose((1, 2, 0))
        if num_channels == 1:
            plt.imshow(generated_images, cmap='gray')
        elif num_channels == 3:
            plt.imshow(generated_images)
        plt.title(f'Generated Images - Epoch {epoch}')
        plt.axis('off')
        plt.show()
        
    def save_model(self, file_path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])