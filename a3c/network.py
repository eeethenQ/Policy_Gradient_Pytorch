import torch
import torch.nn as nn
import torch.nn.functional as F



class Net_Pixel(nn.Module):
    def __init__(self, h, w, outputs):
        super(Net_Pixel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.out_dim = outputs

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # convh = (conv2d_size_out(conv2d_size_out(h)))
        # convw = (conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, self.out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        if self.out_dim == 1:
            return x
        else:
            return F.softmax(x, dim=-1)

class Net_Simple(nn.Module):
    def __init__(self, outputs):
        super(Net_Simple, self).__init__()
        self.out_dim = outputs
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, self.out_dim)  # Prob of Left

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.out_dim == 1:
            x = self.fc3(x)
        else:
            x = F.softmax(self.fc3(x), dim=-1)
        return x
