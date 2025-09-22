"""DCGAN code from https://github.com/kpandey008/dcgan"""

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base_c=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # B x in_channels x H x W
            nn.Conv2d(in_channels, base_c, 4, 2, 1, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(base_c),
            # B x base_c x H//2 x W//2
            nn.Conv2d(base_c, base_c * 2, 4, 2, 1, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(base_c * 2),
            # B x base_c*2 x H//2 x W//2
            nn.Conv2d(base_c * 2, base_c * 4, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(base_c * 4),
            # B x base_c*4 x H//4 x W//4
            nn.Conv2d(base_c * 4, base_c * 8, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # B x base_c*8 x 5 x 5
            nn.Conv2d(base_c * 8, base_c * 8, 3, 1, 0, bias=False),
            # B x base_c*8 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    def __init__(self, in_channels=512, out_channels=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            # B x in_channels x 1 x 1
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels // 2),
            # B x in_channels//2 x 3 x 3
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels // 4),
            # B x in_channels//4 x 5 x 5
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, 1, 0, bias=False),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels // 8),
            # B x in_channels//8 x 7 x 7
            nn.ConvTranspose2d(
                in_channels // 8, in_channels // 16, 4, 2, 1, bias=False
            ),
            nn.Dropout2d(0.1),
            nn.ReLU(True),
            # B x in_channels//16 x 14 x 14
            nn.ConvTranspose2d(in_channels // 16, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Final Output : # B x out_channels x 28 x 28
        )

    def forward(self, input):
        return self.main(input)


class DCGANLikeModel(nn.Module):
    def __init__(self, in_channels=1, base_c=64):
        super(DCGANLikeModel, self).__init__()
        self.discriminator = Discriminator(in_channels=in_channels, base_c=base_c)
        self.generator = Generator(base_c * 8, out_channels=in_channels)

    def forward(self, x):
        return self.generator(self.discriminator(x))
