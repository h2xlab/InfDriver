import torch
import torch.nn as nn
from collections import OrderedDict
from params import par

class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS,self).__init__()
        '''
        DeepVO FlowNetS CNN
        '''
        conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=(7-1)//2)
        leaky_relu_1 = nn.LeakyReLU(0.1)

        conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=(5-1)//2)
        leaky_relu_2 = nn.LeakyReLU(0.1)

        conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=(5-1)//2)
        leaky_relu_3 = nn.LeakyReLU(0.1)

        conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(3-1)//2)
        leaky_relu_3_1 = nn.LeakyReLU(0.1)

        conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=(3-1)//2)
        leaky_relu_4 = nn.LeakyReLU(0.1)

        conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2)
        leaky_relu_4_1 = nn.LeakyReLU(0.1)

        conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=(3-1)//2)
        leaky_relu_5 = nn.LeakyReLU(0.1)

        conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(3-1)//2)
        leaky_relu_5_1 = nn.LeakyReLU(0.1)

        conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=(3-1)//2)
        leaky_relu_6 = nn.LeakyReLU(0.1)

        self.model = nn.Sequential(OrderedDict([
            ('conv1', conv1),
            ('leaky_relu_1', leaky_relu_1),
            ('conv2', conv2),
            ('leaky_relu_2', leaky_relu_2),
            ('conv3', conv3),
            ('leaky_relu_3', leaky_relu_3),
            ('conv3_1', conv3_1),
            ('leaky_relu_3_1', leaky_relu_3_1),
            ('conv4', conv4),
            ('leaky_relu_4', leaky_relu_4),
            ('conv4_1', conv4_1),
            ('leaky_relu_4_1', leaky_relu_4_1),
            ('conv5', conv5),
            ('leaky_relu_5', leaky_relu_5),
            ('conv5_1', conv5_1),
            ('leaky_relu_5_1', leaky_relu_5_1),
            ('conv6', conv6),
            ('leaky_relu_6', leaky_relu_6),
        ]))

    def forward(self, input):

        feature = self.model(input)

        return feature

if __name__ == '__main__':
    dummy_input = torch.randn(8, 6, 384, 640)
    encoder = FlowNetS()
    feature = encoder(dummy_input)
    print(feature.shape)