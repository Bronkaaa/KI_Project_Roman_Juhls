import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
from NeuralNetworkBase import NeuralNetworkBase
import matplotlib.pyplot as plt





def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.constant_(m.bias, 0)



class Generator(nn.Module):
    """ 3 hidden layer generative nn. """
    def __init__(self, z_dim, input_size, hidden):
        super(Generator, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, input_size),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class Discriminator(nn.Module):
    """ 3 hidden layer discriminative nn. """
    def __init__(self, input_size, hidden):
        super(Discriminator, self).__init__()
        
        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x



"""

class Generator(nn.Module):
    def __init__(self, z_dim, input_size, hidden_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, input_size), 
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

"""

class GAN:
    def __init__(self, input_size, z_dim, hidden_size):
        super(GAN, self).__init__()
        
        self.generator = Generator(z_dim, input_size, hidden_size)
        self.discriminator = Discriminator(input_size, hidden_size)

    def train_model(self, data_loader, num_channels, z_dim, input_size, num_epochs, criterion, generator_optimizer, discriminator_optimizer):


        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        g_losses = []
        d_losses = []


        for epoch in range(num_epochs):
            
            self.generator.train()
            
            for real_data, _ in data_loader:
                batch_size = real_data.size(0)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Train Discriminator
                discriminator_optimizer.zero_grad()
                real_data = real_data.view(-1, input_size)
                real_output = self.discriminator(real_data)
                real_loss = criterion(real_output, real_labels)

                noise = torch.randn(batch_size, z_dim)
                generated_data = self.generator(noise)
                
                
                fake_output = self.discriminator(generated_data.detach())
                fake_loss = criterion(fake_output, fake_labels)

                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Train Generator
                generator_optimizer.zero_grad()
                fake_output = self.discriminator(generated_data)
                generator_loss = criterion(fake_output, real_labels)
                generator_loss.backward()
                generator_optimizer.step()
                
            g_losses.append(generator_loss.item())
            d_losses.append(discriminator_loss.item())




            self.plot_images(z_dim)
                              
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Discriminator Loss: {discriminator_loss.item()}, "
                  f"Generator Loss: {generator_loss.item()}")
                   
     
        return g_losses, d_losses
            
        
    def plot_images(self, g_input):
        self.generator.eval()  # Set the generator to evaluation mode
        with torch.no_grad():
            z = torch.randn(16, g_input)
            generated_images = self.generator(z).detach().cpu()

        # Plot the generated images
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_images[i].view(28, 28), cmap='gray')
            ax.axis('off')

        plt.show()   
        
        
        
    def test_model(self):
        return None
            
    def save_model(self, file_path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        