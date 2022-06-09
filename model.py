import torch
import torch.nn as nn
import math


class ResBlockDown(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlockDown, self).__init__()

        self.conv_relu_norm = nn.Sequential(
            nn.Conv3d(inchannel, inchannel, 3, 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(inchannel),

            nn.Conv3d(inchannel, outchannel, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(outchannel),
        )

        self.shortcut = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(outchannel)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_relu_norm(x)
        return residual + x


class ResBlockUp(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlockUp, self).__init__()

        self.conv_relu_norm = nn.Sequential(
            nn.ConvTranspose3d(inchannel, outchannel, 4, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(outchannel),

            nn.Conv3d(outchannel, outchannel, 3, 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(outchannel),
        )

        self.shortcut = nn.Sequential(
            nn.ConvTranspose3d(inchannel, outchannel, 4, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(outchannel)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_relu_norm(x)
        return residual + x


class Encoder(nn.Module):
    def __init__(self, h, w, z, z_dim):
        super(Encoder, self).__init__()

        channel = 16
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, channel, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(channel)
        )

        #  16 > 32 > 64 > 128 > 256
        self.res1 = ResBlockDown(channel, channel * 2)
        self.res2 = ResBlockDown(channel * 2, channel * 4)
        self.res3 = ResBlockDown(channel * 4, channel * 8)
        self.res4 = ResBlockDown(channel * 8, channel * 16)

        self.flat_num = math.ceil(h/16) * math.ceil(w/16) * math.ceil(z/16) * channel * 16
        self.dense = nn.Linear(self.flat_num, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(-1, self.flat_num)
        zcode = self.dense(x)

        return zcode


class Decoder(nn.Module):
    def __init__(self, h, w, z, z_dim):
        super(Decoder, self).__init__()

        self.channel = 16
        self.hshape = math.ceil(h / 16)
        self.wshape = math.ceil(w / 16)
        self.zshape = math.ceil(z / 16)

        self.z_develop = self.hshape * self.wshape * self.zshape * self.channel * 16
        self.dense = nn.Linear(z_dim, self.z_develop)

        self.res4 = ResBlockUp(self.channel * 16, self.channel * 8)
        self.res3 = ResBlockUp(self.channel * 8, self.channel * 4)
        self.res2 = ResBlockUp(self.channel * 4, self.channel * 2)
        self.res1 = ResBlockUp(self.channel * 2, self.channel)

        self.lastconv = nn.Conv3d(self.channel, 1, 1)

    def forward(self, z):
        x = self.dense(z)
        x = x.view(-1, self.channel * 16, self.hshape, self.wshape, self.zshape)
        x = self.res4(x)
        x = self.res3(x)
        x = self.res2(x)
        x = self.res1(x)
        out = self.lastconv(x)

        return out


if __name__ == '__main__':
    from torchsummary import summary
    emodel = Encoder(160, 192, 96, 512)
    dmodel = Decoder(160, 192, 96, 512)
    summary(emodel, (1, 160, 192, 96), device='cpu')
    summary(dmodel, (1, 512), device='cpu')