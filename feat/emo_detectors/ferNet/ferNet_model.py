# https://www.kaggle.com/jcheong0428/facial-emotion-recognition
# Pytorch Implementation of the above algorithm
import torch.nn as nn
import torch


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class LocalConv2dReLU(nn.Module):
    def __init__(
        self,
        local_h_num,
        local_w_num,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation_type="ReLU",
    ):
        super(LocalConv2dReLU, self).__init__()
        self.local_h_num = local_h_num
        self.local_w_num = local_w_num

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(in_channels) for i in range(local_h_num * local_w_num)]
        )

        if activation_type == "ReLU":
            self.relus = nn.ModuleList(
                [nn.ReLU(inplace=True) for i in range(local_h_num * local_w_num)]
            )
        elif activation_type == "PReLU":
            self.relus = nn.ModuleList(
                [nn.PReLU() for i in range(local_h_num * local_w_num)]
            )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                )
                for i in range(local_h_num * local_w_num)
            ]
        )

    def forward(self, x):
        h_splits = torch.split(x, int(x.size(2) / self.local_h_num), 2)

        h_out = []
        for i in range(len(h_splits)):
            start = True
            w_splits = torch.split(
                h_splits[i], int(h_splits[i].size(3) / self.local_w_num), 3
            )
            for j in range(len(w_splits)):
                bn_out = self.bns[i * len(w_splits) + j](w_splits[j].contiguous())
                bn_out = self.relus[i * len(w_splits) + j](bn_out)
                conv_out = self.convs[i * len(w_splits) + j](bn_out)
                if start:
                    h_out.append(conv_out)
                    start = False
                else:
                    h_out[i] = torch.cat((h_out[i], conv_out), 3)
            if i == 0:
                out = h_out[i]
            else:
                out = torch.cat((out, h_out[i]), 2)

        return out


class HierarchicalMultiScaleRegionLayer(nn.Module):
    def __init__(
        self,
        local_group,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation_type="ReLU",
    ):
        super(HierarchicalMultiScaleRegionLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.local_conv_branch1 = LocalConv2dReLU(
            local_group[0][0],
            local_group[0][1],
            out_channels,
            int(out_channels / 2),
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation_type,
        )
        self.local_conv_branch2 = LocalConv2dReLU(
            local_group[1][0],
            local_group[1][1],
            int(out_channels / 2),
            int(out_channels / 4),
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation_type,
        )
        self.local_conv_branch3 = LocalConv2dReLU(
            local_group[2][0],
            local_group[2][1],
            int(out_channels / 4),
            int(out_channels / 4),
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            activation_type,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if activation_type == "ReLU":
            self.relu = nn.ReLU(inplace=True)
        elif activation_type == "PReLU":
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        local_branch1 = self.local_conv_branch1(x)
        local_branch2 = self.local_conv_branch2(local_branch1)
        local_branch3 = self.local_conv_branch3(local_branch2)
        local_out = torch.cat((local_branch1, local_branch2, local_branch3), 1)

        out = x + local_out
        out = self.bn(out)
        out = self.relu(out)

        return out


class fer_net(nn.Module):
    def __init__(self, in_chs, num_classes, img_size=200):
        super(fer_net, self).__init__()

        self.HMR = HierarchicalMultiScaleRegionLayer(
            [[8, 8], [4, 4], [2, 2]],
            in_chs,
            4 * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            activation_type="ReLU",
        )

        self.Conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * 4, out_channels=64, kernel_size=(5, 5), padding=(2, 2)
            ),  # Resulting image shape: (x-2,x-2,64)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2)
            ),  # Resulting image shape: (x-4,x-4,64)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=64),
            # Resulting image shape: (x,x,64)
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)
            ),  # Resulting image shape: (x/2-2,x/2-2,256)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)
            ),
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
            ),
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)
            ),
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.5),
        )

        self.Linear_layers = nn.Sequential(
            nn.Linear(in_features=int(625 * 256), out_features=128),
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.6),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(0.6),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        x = self.HMR(x)
        x = self.Conv_layers(x)
        x = x.view(x.size(0), -1)
        # x = x.flatten()
        outs = self.Linear_layers(x)

        return outs


if __name__ == "__main__":
    A01 = fer_net(in_chs=3, num_classes=7, img_size=200)
    # print(A01)
    rand_data = torch.rand(12, 3, 200, 200)
    ou1 = A01(rand_data)
    # print(ou1.shape)
    # print(summary(A01, input_size = (12, 1, 48, 48)))
