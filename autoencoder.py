"""
A simple autoencoder for the MNIST dataset.
"""

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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        """
        Train the network.
        """
        for epoch in range(2):
            running_loss = 0.0
            for step, examples in enumerate(self.train_data_loader):
                images, labels = examples
                images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
                self.optimizer.zero_grad()
                decoded_images = self.net(images)
                loss = (decoded_images - images).pow(2).sum()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                if step % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 2000))
                    running_loss = 0.0
        

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
        self.fully_connected2 = torch.nn.Linear(120, 84)
        # Decode.
        self.fully_connected3 = torch.nn.Linear(84, 120)
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
        values = torch.nn.functional.relu(self.convolution1(input_tensor))
        values = torch.nn.functional.relu(self.convolution2(values))
        values = values.view(-1, 20 * 5 * 5)
        values = torch.nn.functional.relu(self.fully_connected1(values))
        values = torch.nn.functional.relu(self.fully_connected2(values))
        # Decode.
        values = torch.nn.functional.tanh(self.fully_connected3(values))
        values = torch.nn.functional.tanh(self.fully_connected4(values))
        values = values.view(-1, 20, 5, 5)
        values = torch.nn.functional.tanh(self.transposed_convolution1(values))
        values = self.transposed_convolution2(values)[:, :, 1:, 1:]
        return values


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
    autoencoder = Autoencoder()
    autoencoder.train()
