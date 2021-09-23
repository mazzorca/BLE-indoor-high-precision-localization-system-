import torchvision
import torch

import matplotlib.pyplot as plt
import numpy as np

from net import Net


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join([classes[labels[j]] for j in range(4)]))

    # net = Net()
    # PATH = './cifar_net.pth'
    # net.load_state_dict(torch.load(PATH))
    #
    # outputs = net(images)
    #
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join([classes[predicted[j]] for j in range(4)]))
