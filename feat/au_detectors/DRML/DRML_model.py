# We first did implementation according to original paper. But we
# validated our model with https://github.com/AlexHex7/DRML_pytorch and updated
# our codes accordingly.
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionLayer(nn.Module):
    def __init__(self, in_channels, grid=(8, 8)):
        super(RegionLayer, self).__init__()
        self.in_channels = in_channels
        self.grid = grid
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for _ in range(grid[0] * grid[1])
            ]
        )

    def forward(self, x):
        """
        Shape of input/output : (B,C,H,W)
        """

        _, _, height, width = x.size()

        h_splits = torch.split(x, height // self.grid[0], dim=2)
        outputs = []
        for i, h_chuncks in enumerate(h_splits):

            w_splits = torch.split(h_chuncks, width // self.grid[1], dim=3)
            output_chunk = []
            for j, w_chunks in enumerate(w_splits):
                grid_val0 = self.networks[i * j + j](w_chunks.contiguous()) + w_chunks
                output_chunk.append(grid_val0)
            w_cats = torch.cat(output_chunk, dim=3)
            outputs.append(w_cats)

        return torch.cat(outputs, dim=2)


class DRML_net(nn.Module):
    def __init__(self, AU_num=12):
        super(DRML_net, self).__init__()
        self.AU_Num = 12
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=1),
            RegionLayer(in_channels=32, grid=(8, 8)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(in_features=16 * 27 * 27, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=AU_num * 2),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.extractor(x)
        x1 = x1.view(batch_size, -1)
        x1 = self.linear_classifier(x1)
        x1 = x1.view(batch_size, 2, self.AU_Num)
        x1 = F.log_softmax(x1, dim=1)
        return x1


if __name__ == "__main__":
    from torch.autograd import Variable

    pic_x = Variable(torch.rand(8, 3, 170, 170))
    net = DRML_net(AU_num=12)
    print(net)
    result01 = net(pic_x)
    print(result01)
    print("finished")
