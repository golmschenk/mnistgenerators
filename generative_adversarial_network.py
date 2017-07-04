"""
Code for a simple generative adversarial network for the MNIST data.
"""
import os

import shutil
import torch
import torchvision


class GenerativeAdversarialNetwork:
    """
    A class for a simple generative adversarial network for the MNIST data.
    """
    def __init__(self):
        data_loaders = create_mnist_data_loaders()


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
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                    shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=data_preprocessing_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                                   shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


class Discriminator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """
    def __init__(self):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=1)
        self.convolution2 = torch.nn.Conv2d(10, 20, kernel_size=5, stride=2)
        self.fully_connected1 = torch.nn.Linear(20 * 5 * 5, 120)
        self.fully_connected2 = torch.nn.Linear(120, 60)
        self.fully_connected2 = torch.nn.Linear(60, 10)


class Generator(torch.nn.Module):
    """
    The class for the discriminator network structure.
    """
    def __init__(self):
        super().__init__()
        self.fully_connected1 = torch.nn.Linear(60, 120)
        self.fully_connected2 = torch.nn.Linear(120, 20 * 5 * 5)
        self.transposed_convolution1 = torch.nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2)
        self.transposed_convolution2 = torch.nn.ConvTranspose2d(10, 1, kernel_size=5, stride=2)


def reset_results_directory():
    shutil.rmtree('results')
    os.makedirs('results')


if __name__ == '__main__':
    reset_results_directory()
    gan = GenerativeAdversarialNetwork()
    gan.train()