"""
A simple autoencoder for the MNIST dataset.
"""

import os
import numpy as np
import scipy.misc
import shutil
import torch
import torchvision


class Autoencoder:
    """
    A class for an autoencoder for the MNIST dataset.
    """
    def __init__(self):
        data_loaders = create_mnist_data_loaders()
        self.train_data_loader = data_loaders[0]
        self.test_data_loader = data_loaders[1]
        self.data_classes = tuple(map(str, list(range(10))))
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001,)
        self.latent_loss = None

    def train(self):
        """
        Train the network.
        """
        for epoch in range(10):
            running_loss = 0.0
            for step, examples in enumerate(self.train_data_loader):
                images, labels = examples
                images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
                self.optimizer.zero_grad()
                decoded_images = self.net(images)
                loss = (decoded_images - images).pow(2).sum() + 3 * self.net.latent_loss.sum()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                if step % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 2000))
                    running_loss = 0.0
                    file_name = 'epoch_{}_step_{}'.format(epoch, step)
                    save_original_decoded_pair_to_image_file(images[0], decoded_images[0], file_name)
                    generated_example = self.net.generate_example()
                    scipy.misc.imsave(os.path.join('results', 'generated_epoch_{}_step_{}.jpg'.format(epoch, step)),
                                      generated_example[0][0].data.numpy())



def reset_results_directory():
    shutil.rmtree('results')
    os.makedirs('results')

def save_original_decoded_pair_to_image_file(original_image_variable, decoded_image_variable, file_name):
    original_image = original_image_variable[0].data.numpy()
    decoded_image = decoded_image_variable[0].data.numpy()
    combined_image = np.concatenate([original_image, decoded_image], axis=1)
    scipy.misc.imsave(os.path.join('results', '{}.jpg'.format(file_name)), combined_image)


class Net(torch.nn.Module):
    """
    The class for the network structure.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Encode.
        self.convolution1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=2, padding=1)
        self.convolution2 = torch.nn.Conv2d(10, 20, kernel_size=5, stride=2)
        self.fully_connected1 = torch.nn.Linear(20 * 5 * 5, 120)
        self.fully_connected2_mean = torch.nn.Linear(120, 70)
        self.fully_connected2_stddev = torch.nn.Linear(120, 70)
        # Decode.
        self.fully_connected3 = torch.nn.Linear(70, 120)
        self.fully_connected4 = torch.nn.Linear(120, 20 * 5 * 5)
        self.transposed_convolution1 = torch.nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2)
        self.transposed_convolution2 = torch.nn.ConvTranspose2d(10, 1, kernel_size=5, stride=2)

    def forward(self, input_tensor):
        """
        Performs the forward pass of the network.

        :param input_tensor: The input tensor to the inference.
        :type input_tensor: torch.Tensor
        :return: The resulting inferred tensor.
        :rtype: torch.Tensor
        """
        # Encode.
        mean_values, stddev_values = self.encode(input_tensor)
        stddev_values = stddev_values.abs() + 0.0000001
        self.latent_loss = 0.5 * (mean_values.pow(2) + stddev_values.pow(2) - stddev_values.pow(2).log() - 1).sum(1)
        normal_sample = torch.autograd.Variable(torch.randn([4, 70]), requires_grad=False)
        latent_samples = mean_values + (stddev_values * normal_sample)
        # Decode.
        values = self.decode(latent_samples)
        return values

    def encode(self, input_tensor):
        values = torch.nn.functional.relu(self.convolution1(input_tensor))
        values = torch.nn.functional.relu(self.convolution2(values))
        values = values.view(-1, 20 * 5 * 5)
        values = torch.nn.functional.relu(self.fully_connected1(values))
        mean_values = torch.nn.functional.relu(self.fully_connected2_mean(values))
        stddev_values = torch.nn.functional.relu(self.fully_connected2_stddev(values))
        return mean_values, stddev_values

    def decode(self, latent_samples):
        values = torch.nn.functional.tanh(self.fully_connected3(latent_samples))
        values = torch.nn.functional.tanh(self.fully_connected4(values))
        values = values.view(-1, 20, 5, 5)
        values = torch.nn.functional.tanh(self.transposed_convolution1(values))
        values = self.transposed_convolution2(values)[:, :, 1:, 1:]
        return values

    def generate_example(self):
        normal_sample = torch.autograd.Variable(torch.randn([1, 70]), requires_grad=False)
        return self.decode(normal_sample)


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


if __name__ == '__main__':
    reset_results_directory()
    autoencoder = Autoencoder()
    autoencoder.train()
