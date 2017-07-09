"""
The tutorial code.
"""

import numpy as np
import torch
import torchvision

if __name__ == '__main__':

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

    classes = tuple(list(range(10)))


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
            self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
            self.fc2 = torch.nn.Linear(120, 84)
            self.fc3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = self.pool(torch.nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    for data in test_data_loader:
        images, labels = data
        outputs = net(torch.autograd.Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))