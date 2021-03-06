""" SqueezeSeg Model """

import torch
import torch.nn as nn
import torch.nn.functional as F

# from .bilateral_filter import BilateralFilter
# from .recurrent_crf import RecurrentCRF
import IPython


class Conv(nn.Module):
    def __init__(self, inputs, outputs, kernel_size=3, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outputs)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MaxPool(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True)

    def forward(self, x):
        return self.pool(x)


class Fire(nn.Module):
    def __init__(self, inputs, o_sq1x1, o_ex1x1, o_ex3x3):
        """ Fire layer constructor.
        Args:
            inputs : input tensor
            o_sq1x1 : output of squeeze layer
            o_ex1x1 : output of expand layer(1x1)
            o_ex3x3 : output of expand layer(3x3)
        """
        super(Fire, self).__init__()
        self.sq1x1 = Conv(inputs, o_sq1x1, kernel_size=1, stride=1, padding=0)
        self.ex1x1 = Conv(o_sq1x1, o_ex1x1, kernel_size=1, stride=1, padding=0)
        self.ex3x3 = Conv(o_sq1x1, o_ex3x3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return torch.cat([self.ex1x1(self.sq1x1(x)), self.ex3x3(self.sq1x1(x))], 1)


class Deconv(nn.Module):
    def __init__(self, inputs, outputs, kernel_size, stride, padding=0):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outputs)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)))


class FireDeconv(nn.Module):
    def __init__(self, inputs, o_sq1x1, o_ex1x1, o_ex3x3, stride=[1, 2], padding=[0, 1]):
        super(FireDeconv, self).__init__()
        self.sq1x1 = Conv(inputs, o_sq1x1, 1, 1, 0)
        self.deconv = Deconv(o_sq1x1, o_sq1x1, [1, 4], stride, padding)
        self.ex1x1 = Conv(o_sq1x1, o_ex1x1, 1, 1, 0)
        self.ex3x3 = Conv(o_sq1x1, o_ex3x3, 3, 1, 1)

    def forward(self, x):
        x = self.sq1x1(x)
        x = self.deconv(x)
        return torch.cat([self.ex1x1(x), self.ex3x3(x)], 1)


class SqueezeRefine(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SqueezeRefine, self).__init__()
        # encoder
        self.conv1 = Conv(in_channels, 64, 3, (1, 2), 1)
        self.conv1_skip = Conv(in_channels, 64, 1, 1, 0)
        self.pool1 = MaxPool(3, (1, 2), (1, 0))

        self.fire2 = Fire(64, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = MaxPool(3, (1, 2), (1, 0))

        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.pool5 = MaxPool(3, (1, 2), (1, 0))

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        # decoder
        self.fire10 = FireDeconv(512, 64, 128, 128)
        self.fire11 = FireDeconv(256, 32, 64, 64)
        self.fire12 = FireDeconv(128, 16, 32, 32)
        self.fire13 = FireDeconv(64, 16, 32, 32)
        self.feats_channels = 64
        # self.drop = nn.Dropout2d()

        # self.out = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

        # self.out = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
        # )
        # self.out = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
        # )

    def forward(self, x, skip_feats=None):
        feats_list = []
        # [2, 9, 64, 2048]
        out_c1 = self.conv1(x)  # [2, 64, 64, 1024]
        out = self.pool1(out_c1)  # [2, 64, 64, 512]

        out_f3 = self.fire3(self.fire2(out))  # [2, 128, 64, 512]
        out = self.pool3(out_f3)  # [2, 128, 64, 256]

        out_f5 = self.fire5(self.fire4(out))  # [2, 256, 64, 256]
        out = self.pool5(out_f5)  # [2, 256, 64, 128]

        out = self.fire9(self.fire8(self.fire7(self.fire6(out))))  # [2, 512, 64, 128]

        # decoder
        out = torch.add(self.fire10(out), out_f5)  # [2, 256, 64, 256]
        if skip_feats is not None:
            out = torch.add(out, skip_feats[0])
        feats_list.append(out)
        out = torch.add(self.fire11(out), out_f3)  # [2, 128, 64, 512]
        if skip_feats is not None:
            out = torch.add(out, skip_feats[1])
        feats_list.append(out)
        out = torch.add(self.fire12(out), out_c1)  # [2, 64, 64, 1024]
        if skip_feats is not None:
            out = torch.add(out, skip_feats[2])
        feats_list.append(out)
        # out = self.drop(torch.add(self.fire13(out), self.conv1_skip(x)))
        out = torch.add(self.fire13(out), self.conv1_skip(x))  # [2, 64, 64, 2048]
        if skip_feats is not None:
            out = torch.add(out, skip_feats[3])
        feats_list.append(out)
        # out = self.conv14(out)
        # out = self.out(feats)
        # return out, feats
        return feats_list


# class SqueezeRefine(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(SqueezeRefine, self).__init__()
#         # encoder
#         self.conv1 = Conv(in_channels, 64, 3, (1, 2), 1)
#         self.conv1_skip = Conv(in_channels, 64, 1, 1, 0)
#         self.pool1 = MaxPool((3, 5), (1, 4), (1, 0))
#
#         self.fire2 = Fire(64, 16, 64, 64)
#         # self.fire3 = Fire(128, 16, 64, 64)
#         self.pool3 = MaxPool((3, 5), (1, 4), (1, 0))
#
#         self.fire4 = Fire(128, 32, 128, 128)
#         # self.fire5 = Fire(256, 32, 128, 128)
#         self.pool5 = MaxPool(3, (1, 2), (1, 0))
#
#         self.fire6 = Fire(256, 48, 192, 192)
#         # self.fire7 = Fire(384, 48, 192, 192)
#         self.fire8 = Fire(384, 64, 256, 256)
#         # self.fire9 = Fire(512, 64, 256, 256)
#
#         # decoder
#         self.fire10 = FireDeconv(512, 64, 128, 128)
#         self.fire11 = FireDeconv(256, 32, 64, 64, [1, 4], [0, 0])
#         self.fire12 = FireDeconv(128, 16, 32, 32, [1, 4], [0, 0])
#         self.fire13 = FireDeconv(64, 16, 32, 32)
#         self.feats_channels = 64
#         # self.drop = nn.Dropout2d()
#
#         # self.out = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
#
#         # self.out = nn.Sequential(
#         #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
#         # )
#         # self.out = nn.Sequential(
#         #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
#         # )
#
#     def forward(self, x):
#         # [2, 9, 64, 2048]
#         # print('in squeezesegnet:')
#         out_c1 = self.conv1(x)  # [2, 64, 64, 1024]
#         # print('1 ', out_c1.shape)
#         out = self.pool1(out_c1)  # [2, 64, 64, 512]
#         # print('2 ', out.shape)
#
#         out_f3 = self.fire2(out)  # [2, 128, 64, 512]
#         out = self.pool3(out_f3)  # [2, 128, 64, 256]
#         # print('3 ', out.shape)
#
#         out_f5 = self.fire4(out)  # [2, 256, 64, 256]
#         out = self.pool5(out_f5)  # [2, 256, 64, 128]
#         # print('4 ', out.shape)
#
#         # out = self.fire9(self.fire8(self.fire7(self.fire6(out))))  # [2, 512, 64, 128]
#         out = self.fire8(self.fire6(out))  # [2, 512, 64, 128]
#
#         # decoder
#         out = torch.add(self.fire10(out), out_f5)  # [2, 256, 64, 256]
#         # print('5 ', out.shape)
#         out = torch.add(self.fire11(out), out_f3)  # [2, 128, 64, 512]
#         # print('6 ', out.shape)
#         out = torch.add(self.fire12(out), out_c1)  # [2, 64, 64, 1024]
#         # print('7 ', out.shape)
#         # out = self.drop(torch.add(self.fire13(out), self.conv1_skip(x)))
#         feats = torch.add(self.fire13(out), self.conv1_skip(x))  # [2, 64, 64, 2048]
#         # print(out.shape)
#         # out = self.conv14(out)
#         # out = self.out(feats)
#         # return out, feats
#         return feats


class SqueezeSeg(nn.Module):
    # __init__(??????)?????????????????? drop???????????????
    def __init__(self, mc):
        super(SqueezeSeg, self).__init__()

        # config
        self.mc = mc

        # encoder
        self.conv1 = Conv(5, 64, 3, (1, 2), 1)
        self.conv1_skip = Conv(5, 64, 1, 1, 0)
        self.pool1 = MaxPool(3, (1, 2), (1, 0))

        self.fire2 = Fire(64, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = MaxPool(3, (1, 2), (1, 0))

        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.pool5 = MaxPool(3, (1, 2), (1, 0))

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        # decoder
        self.fire10 = FireDeconv(512, 64, 128, 128)
        self.fire11 = FireDeconv(256, 32, 64, 64)
        self.fire12 = FireDeconv(128, 16, 32, 32)
        self.fire13 = FireDeconv(64, 16, 32, 32)

        self.drop = nn.Dropout2d()

        # relu?????????????????????
        self.conv14 = nn.Conv2d(64, mc.NUM_CLASS, kernel_size=3, stride=1, padding=1)

        self.bf = BilateralFilter(mc, stride=1, padding=(1, 2))

        self.rc = RecurrentCRF(mc, stride=1, padding=(1, 2))

    def forward(self, x, lidar_mask):
        # encoder
        out_c1 = self.conv1(x)
        out = self.pool1(out_c1)

        out_f3 = self.fire3(self.fire2(out))
        out = self.pool3(out_f3)

        out_f5 = self.fire5(self.fire4(out))
        out = self.pool5(out_f5)

        out = self.fire9(self.fire8(self.fire7(self.fire6(out))))

        # decoder
        out = torch.add(self.fire10(out), out_f5)
        out = torch.add(self.fire11(out), out_f3)
        out = torch.add(self.fire12(out), out_c1)
        out = self.drop(torch.add(self.fire13(out), self.conv1_skip(x)))
        out = self.conv14(out)

        bf_w = self.bf(x[:, :3, :, :])

        out = self.rc(out, lidar_mask, bf_w)

        return out


class SqueezeSegGenerator(nn.Module):
    def __init__(self, channels):
        super(SqueezeSegGenerator, self).__init__()

        # encoder
        self.conv1 = Conv(channels, 64, 3, (1, 2), 1)
        self.conv1_skip = Conv(channels, 64, 1, 1, 0)
        self.pool1 = MaxPool(3, (1, 2), (1, 0))

        self.fire2 = Fire(64, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = MaxPool(3, (1, 2), (1, 0))

        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.pool5 = MaxPool(3, (1, 2), (1, 0))

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        # decoder
        self.fire10 = FireDeconv(512, 64, 128, 128)
        self.fire11 = FireDeconv(256, 32, 64, 64)
        self.fire12 = FireDeconv(128, 16, 32, 32)
        self.fire13 = FireDeconv(64, 16, 32, 32)

        self.drop = nn.Dropout2d()

        # relu?????????????????????
        self.conv14 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        # encoder
        out_c1 = self.conv1(x)
        out = self.pool1(out_c1)

        out_f3 = self.fire3(self.fire2(out))
        out = self.pool3(out_f3)

        out_f5 = self.fire5(self.fire4(out))
        out = self.pool5(out_f5)

        out = self.fire9(self.fire8(self.fire7(self.fire6(out))))

        # decoder
        out = torch.add(self.fire10(out), out_f5)
        out = torch.add(self.fire11(out), out_f3)
        out = torch.add(self.fire12(out), out_c1)
        out = self.drop(torch.add(self.fire13(out), self.conv1_skip(x)))
        out = self.conv14(out)

        return out.permute((0, 2, 3, 1))