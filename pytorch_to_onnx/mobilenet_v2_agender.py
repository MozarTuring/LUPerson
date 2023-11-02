# coding: utf-8
"""
https://pytorch.org/docs/1.6.0/_modules/torchvision/models/mobilenet.html#mobilenet_v2
"""
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Agender(nn.Module):
    def __init__(self, width_mult=1.0, last_channel=1280, gender_max=1, age_max=100, pth_path=None):
        super(MobileNetV2Agender, self).__init__()
        self.age_max = age_max

        input_channel = 32
        round_nearest = 8
        block = InvertedResidual
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f'inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}')

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building heads
        self.gender = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, gender_max + 1),
        )

        self.age = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, 1 + self.age_max),
            nn.Dropout(0.2),
            nn.Linear(1 + self.age_max, 1),
            nn.Sigmoid(),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if pth_path is not None:
            state_dict = torch.load(pth_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x, targets=None):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        gender = self.gender(x)
        age = self.age(x).squeeze(dim=1) * self.age_max

        if targets is None:
            return {
                'gender': gender,
                'age': age,
            }
        else:
            return {
                'gender': (gender, targets['gender']),
                'age': (age, targets['age']),
            }


if __name__ == '__main__':
    net = MobileNetV2Agender(width_mult=1.0, last_channel=1280, gender_max=1, age_max=100, pth_path=None)
    if torch.cuda.is_available():
        net.cuda()

    torch.save(net.state_dict(), 'mobilenet_v2_agender_test.pth')

    repeat = 1000
    time_list = []
    for i in tqdm(range(repeat)):
        inputs = torch.rand(1, 3, 112, 96)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        t_start = time.time()
        outputs = net(inputs)
        t_end = time.time() - t_start
        time_list.append(t_end)

    time_list = time_list[10:]  # 计算平均值去掉最开始10个
    print(f'\ntime | median={np.median(time_list) * 1000:.1f}ms | mean={np.mean(time_list) * 1000:.1f}ms | '
          f'max={np.max(time_list) * 1000:.1f}ms | min={np.min(time_list) * 1000:.1f}ms')

    print('\nfinished!')
