import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm

from .utils import calc_padding

SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlock, self).__init__()

        self.C = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(SLOPE),
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=dilations[i // 2][i % 2],
                        padding=calc_padding(kernel_size, dilations[i // 2][i % 2])
                    )
                )
            )
            for i in range(len(dilations) * 2)
        ])

        for seq in self.C:
            for module in seq:
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, 0., 0.01)

    def forward(self, x):
        for i, module in enumerate(self.C):
            if i % 2 == 0:
                residual = x
            x = module(x)
            if i % 2 == 1:
                x += residual
        return x

    def remove_weight_norm(self):
        for module in self.C.children():
            if hasattr(module, 'weight'):
                remove_weight_norm(module)


class HiFiGenerator(nn.Module):
    def __init__(self, channels=128, up_kernel_sizes=[16, 16, 4, 4],
                 conv_kernel_sizes=[3, 7, 11], dilations=[[1, 1], [3, 1], [5, 1]]):
        super(HiFiGenerator, self).__init__()

        self.num_res_blocks = len(conv_kernel_sizes)

        self.IN_C = weight_norm(
            nn.Conv1d(
                in_channels=80,
                out_channels=channels,
                kernel_size=7,
                padding=7 // 2
            )
        )

        self.UP = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(SLOPE),
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels=channels // 2 ** i,
                        out_channels=channels // 2 ** (i + 1),
                        kernel_size=up_kernel_sizes[i],
                        stride=up_kernel_sizes[i] // 2,
                        padding=(up_kernel_sizes[i] - up_kernel_sizes[i] // 2) // 2,
                    )
                )
            )
            for i in range(len(up_kernel_sizes))
        ])

        self.RES = nn.ModuleList([
            nn.ModuleList([
                ResBlock(
                    channels=channels // 2 ** (i + 1),
                    kernel_size=conv_kernel_sizes[j],
                    dilations=dilations
                )
                for j in range(len(conv_kernel_sizes))
            ])
            for i in range(len(up_kernel_sizes))
        ])

        self.OUT = nn.Sequential(
            nn.LeakyReLU(SLOPE),
            weight_norm(
                nn.Conv1d(
                    in_channels=channels // 2 ** (len(up_kernel_sizes)),
                    out_channels=1,
                    kernel_size=7,
                    padding=7 // 2
                )
            ),
            nn.Tanh()
        )

        for seq in self.UP:
            for module in seq:
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, 0., 0.01)

        for module in self.OUT:
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, 0., 0.01)

    def forward(self, x):
        x = self.IN_C(x)
        for i, module in enumerate(self.UP):
            x = module(x)
            for j, resblock in enumerate(self.RES[i]):
                if j == 0:
                    sum_x = resblock(x)
                else:
                    sum_x += resblock(x)
            x = sum_x / self.num_res_blocks
        x = self.OUT(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.IN_C)

        for seq in self.UP:
            for module in seq:
                if hasattr(module, 'weight'):
                    remove_weight_norm(module)

        for mlist in self.RES:
            for resblock in mlist:
                resblock.remove_weight_norm()

        for module in self.OUT:
            if hasattr(module, 'weight'):
                remove_weight_norm(module)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()

        CHANNELS = [32, 128, 512, 1024, 1024]

        self.period = period
        self.C = nn.ModuleList([
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_channels=1 if i == 0 else CHANNELS[i - 1],
                        out_channels=CHANNELS[i],
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(calc_padding(5, 1), 0)
                    )
                ),
                nn.LeakyReLU(SLOPE)
            )
            for i in range(len(CHANNELS))
        ])

        self.C_OUT = weight_norm(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(3, 1),
                padding=(1, 0)
            )
        )

        self.F = nn.Flatten()

    def forward(self, x):
      bsz, _, time = x.shape
      feature_maps = []

      # padding
      if time % self.period != 0:
          to_pad = self.period - time % self.period
          x = torch.nn.functional.pad(x, (0, to_pad), 'reflect')
          time += to_pad
      x = x.view(bsz, 1, time // self.period, self.period).contiguous()

      for module in self.C:
          x = module(x)
          feature_maps.append(x)

      x = self.C_OUT(x)
      feature_maps.append(x)
      x = self.F(x)
      return x, feature_maps


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()

        CHANNELS = [128, 128, 256, 512, 1024, 1024, 1024]
        GROUPS = [1, 4, 16, 16, 16, 16, 1]
        KERNEL_SIZES = [15, 41, 41, 41, 41, 41, 5]
        STRIDES = [1, 2, 2, 4, 4, 1, 1]


        self.C = nn.ModuleList([
            nn.Sequential(
                weight_norm(
                    nn.Conv1d(
                        in_channels=1 if i == 0 else CHANNELS[i - 1],
                        out_channels=CHANNELS[i],
                        kernel_size=KERNEL_SIZES[i],
                        stride=STRIDES[i],
                        padding=KERNEL_SIZES[i] // 2,
                        groups=GROUPS[i]
                    )
                ),
                nn.LeakyReLU(SLOPE)
            )
            for i in range(len(CHANNELS))
        ])

        self.C_OUT = weight_norm(
            nn.Conv1d(
                in_channels=1024,
                out_channels=1,
                kernel_size=3,
                padding=1
            )
        )

        self.F = nn.Flatten()

    def forward(self, x):
      feature_maps = []

      for module in self.C:
          x = module(x)
          feature_maps.append(x)

      x = self.C_OUT(x)
      feature_maps.append(x)
      x = self.F(x)
      return x, feature_maps


class MPD(nn.Module):
    def __init__(self):
        super(MPD, self).__init__()
        PERIODS = [2, 3, 5, 7, 11]

        self.PDs = nn.ModuleList([
            PeriodDiscriminator(period)
            for period in PERIODS
        ])

    def forward(self, x):
        feature_maps = []
        preds = []
        for discriminator in self.PDs:
            pred, fmp = discriminator(x)
            feature_maps += fmp
            preds.append(pred)
        return preds, feature_maps


class MSD(nn.Module):
    def __init__(self):
        super(MSD, self).__init__()

        self.P = nn.AvgPool1d(
            kernel_size=4,
            stride=2,
            padding=2
        )

        self.SDs = nn.ModuleList([
            ScaleDiscriminator()
            for _ in range(3)
        ])

    def forward(self, x):
        feature_maps = []
        preds = []
        for i, discriminator in enumerate(self.SDs):
            if i != 0:
                x = self.P(x)
            pred, fmp = discriminator(x)
            feature_maps += fmp
            preds.append(pred)
        return preds, feature_maps


class HiFiDiscriminator(nn.Module):
    def __init__(self):
        super(HiFiDiscriminator, self).__init__()

        self.MPD = MPD()
        self.MSD = MSD()

    def forward(self, y_gen, y_true):
        gen_period = self.MPD(y_gen)
        gen_scale = self.MSD(y_gen)

        true_period = self.MPD(y_true)
        true_scale = self.MSD(y_true)

        preds_generated = gen_period[0] + gen_scale[0]
        preds_true = true_period[0] + true_scale[0]

        fmaps_generated = gen_period[1] + gen_scale[1]
        fmaps_true = true_period[1] + true_scale[1]
        return preds_generated, preds_true, fmaps_generated, fmaps_true
