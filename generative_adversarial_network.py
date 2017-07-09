"""
Code for a simple generative adversarial network for the MNIST data.
"""
import os
import itertools
import shutil
import scipy.misc
import torch
import torchvision

from utility import make_this_process_low_priority


class GenerativeAdversarialNetwork:
    """
    A class for a simple generative adversarial network for the MNIST data.
    """
    def __init__(self):
        data_loaders = create_mnist_data_loaders()
        self.train_data_loader = data_loaders[0]
        self.test_data_loader = data_loaders[1]
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.001)
        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=0.001)

    def train(self):
        """
        Train the network.
        """
        for epoch in range(20):
            discriminator_running_loss = 0.0
            generator_running_loss = 0.0
            for step, examples in enumerate(self.train_data_loader):
                images, _ = examples
                images = torch.autograd.Variable(images)
                normal_samples = torch.autograd.Variable(torch.randn([10, 100]))
                generated_images = self.generator(normal_samples)
                real_scores = self.discriminator(images)
                fake_scores = self.discriminator(generated_images)
                discriminator_loss = fake_scores.mean() - real_scores.mean()
                generator_loss = -fake_scores.mean()
                self.discriminator_optimizer.zero_grad()
                retain_variables = step % 10 == 0
                discriminator_loss.backward(retain_variables=retain_variables)
                self.discriminator_optimizer.step()
                for parameter in self.discriminator.parameters():
                    parameter.data.clamp_(-1.0, 1.0)
                if step % 10 == 0:
                    self.generator_optimizer.zero_grad()
                    generator_loss.backward()
                    self.generator_optimizer.step()
                discriminator_running_loss += discriminator_loss.data[0]
                generator_running_loss += generator_loss.data[0]
                if step % 1000 == 999:
                    print('[{:d}, {:5d}] Generator Loss: {:.3f}, Discriminator Loss {:.3f}'
                          .format(epoch, step, generator_running_loss / 1000,
                                  discriminator_running_loss / 1000))
                    if len(generated_images.size()) == 2:
                        generated_images = generated_images.view(-1, 1, 28, 28)
                    scipy.misc.imsave(os.path.join('results', 'generated_epoch_{}_step_{}.jpg'.format(epoch, step)),
                                      generated_images[0][0].data.numpy())
                    discriminator_running_loss = 0.0
                    generator_running_loss = 0.0


def create_mnist_data_loaders():
    """
    Creates the data loaders for the MNIST dataset.

    :return: The train and test data loaders.
    :rtype: (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    data_preprocessing_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=data_preprocessing_transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10,
                                                    shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=data_preprocessing_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10,
                                                   shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


class Discriminator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """
    def __init__(self):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(1, 30, kernel_size=5, stride=2, padding=1)
        self.convolution2 = torch.nn.Conv2d(30, 60, kernel_size=5, stride=2, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm2d(60)
        self.fully_connected1 = torch.nn.Linear(60 * 5 * 5, 60, bias=False)
        self.batch_norm3 = torch.nn.BatchNorm1d(60)
        self.fully_connected2 = torch.nn.Linear(60, 1)

    def forward(self, input_tensor):
        """
        Forward pass of the discriminator.

        :param input_tensor: The image to discriminate.
        :type input_tensor: torch.autograd.Variable
        :return: The scores of being a real or fake image.
        :rtype: torch.autograd.Variable
        """
        values = torch.nn.functional.leaky_relu(self.convolution1(input_tensor))
        values = torch.nn.functional.leaky_relu(self.batch_norm2(self.convolution2(values)))
        values = values.view(-1, 60 * 5 * 5)
        values = torch.nn.functional.leaky_relu(self.batch_norm3(self.fully_connected1(values)))
        values = self.fully_connected2(values)
        return values

class SimpleDiscriminator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """

    def __init__(self):
        super().__init__()
        self.fully_connected1 = torch.nn.Linear(784, 128)
        self.fully_connected2 = torch.nn.Linear(128, 1)

    def forward(self, input_tensor):
        """
        Forward pass of the discriminator.

        :param input_tensor: The image to discriminate.
        :type input_tensor: torch.autograd.Variable
        :return: The scores of being a real or fake image.
        :rtype: torch.autograd.Variable
        """
        values = input_tensor.contiguous()
        values = values.view(-1, 784)
        values = torch.nn.functional.leaky_relu(self.fully_connected1(values))
        values = self.fully_connected2(values)
        return values


class Generator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """
    def __init__(self):
        super().__init__()
        self.fully_connected1 = torch.nn.Linear(100, 120, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(120)
        self.transposed_convolution1 = torch.nn.ConvTranspose2d(120, 60, kernel_size=5, bias=False)
        self.batch_norm2 = torch.nn.BatchNorm2d(60)
        self.transposed_convolution2 = torch.nn.ConvTranspose2d(60, 30, kernel_size=5, stride=2, bias=False)
        self.batch_norm3 = torch.nn.BatchNorm2d(30)
        self.transposed_convolution3 = torch.nn.ConvTranspose2d(30, 1, kernel_size=5, stride=2)

    def forward(self, input_tensor):
        """
        Forward pass of the generator.

        :param input_tensor: The latent noise tensor to generate an image from.
        :type input_tensor: torch.autograd.Variable
        :return: The generated image.
        :rtype: torch.autograd.Variable
        """
        values = torch.nn.functional.leaky_relu(self.batch_norm1(self.fully_connected1(input_tensor)))
        values = values.view(-1, 120, 1, 1)
        values = torch.nn.functional.leaky_relu(self.batch_norm2(self.transposed_convolution1(values)))
        values = torch.nn.functional.leaky_relu(self.batch_norm3(self.transposed_convolution2(values)))
        values = torch.nn.functional.tanh(self.transposed_convolution3(values))
        return values[:, :, 1:, 1:]

class SimpleGenerator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """

    def __init__(self):
        super().__init__()
        self.fully_connected1 = torch.nn.Linear(100, 128, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        self.fully_connected2 = torch.nn.Linear(128, 784)

    def forward(self, input_tensor):
        """
        Forward pass of the generator.

        :param input_tensor: The latent noise tensor to generate an image from.
        :type input_tensor: torch.autograd.Variable
        :return: The generated image.
        :rtype: torch.autograd.Variable
        """
        values = torch.nn.functional.leaky_relu(self.batch_norm1(self.fully_connected1(input_tensor)))
        values = torch.nn.functional.tanh(self.fully_connected2(values))
        return values


def reset_results_directory():
    shutil.rmtree('results')
    os.makedirs('results')


if __name__ == '__main__':
    make_this_process_low_priority()
    reset_results_directory()
    gan = GenerativeAdversarialNetwork()
    gan.train()
