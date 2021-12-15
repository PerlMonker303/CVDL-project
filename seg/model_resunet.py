import torch
import torch.nn as nn
from utils import inference

USE_BN = True


class BatchNormRelu(nn.Module):
    def __init__(self, in_channels):
        super(BatchNormRelu, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if USE_BN:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        '''Convolutional layer'''
        self.b1 = BatchNormRelu(in_channels)
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.b2 = BatchNormRelu(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        '''Shortcut Connection (Identity Mapping)'''
        self.s = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        x_input = x
        if USE_BN:
            x = self.b1(x)
        x = self.c1(x)
        if USE_BN:
            x = self.b2(x)
        x = self.c2(x)
        shortcut = self.s(x_input)

        skip = x + shortcut
        return skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = ResidualBlock(in_channels + out_channels, out_channels)

    def forward(self, x_input, skip):
        x = self.upsample(x_input)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()

        '''Encoder 1'''
        self.cl1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.br1 = BatchNormRelu(64)
        self.cl2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cl3 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)

        '''Encoder 2 and 3'''
        self.r2 = ResidualBlock(64, 128, stride=2)
        self.r3 = ResidualBlock(128, 256, stride=2)

        '''Bridge'''
        self.r4 = ResidualBlock(256, 512, stride=2)

        '''Decoder'''
        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)

        '''Output'''
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        '''Encoder 1'''
        x = self.cl1(x_input)
        if USE_BN:
            x = self.br1(x)
        x = self.cl2(x)
        shortcut = self.cl3(x_input)
        skip1 = x + shortcut

        '''Encoder 2 and 3'''
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        '''Bridge'''
        b = self.r4(skip3)

        '''Decoder'''
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        '''Output'''
        output = self.final_conv(d3)

        return self.sigmoid(output)
        # return output


def test():
    x = torch.randn((4, 1, 200, 200)) # bs, nr_channels, w, h
    model = ResUNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
    model = ResUNet(in_channels=3, out_channels=1)
    elapsed = inference(model, image_dim=(224,224))
    print(f'[Elapsed time: {elapsed}]')
