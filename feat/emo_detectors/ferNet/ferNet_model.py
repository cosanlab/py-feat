# %%
# https://www.kaggle.com/jcheong0428/facial-emotion-recognition
# Pytorch Implementation of the above algorithm
import torch.nn as nn
import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
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
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
        
class fer_net(nn.Module):
    def __init__(self, in_chs, num_classes, img_size=48):
        super(fer_net, self).__init__()
        
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=64,
                      kernel_size=(5, 5), padding=(2,2)),  # Resulting image shape: (x-2,x-2,64)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(5, 5), padding = (2,2)),  # Resulting image shape: (x-4,x-4,64)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=64),
            # Resulting image shape: (x,x,64)
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=(1,1)),  # Resulting image shape: (x/2-2,x/2-2,256)
            nn.ELU(alpha=1, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3,3),padding=(1,1)),
            nn.ELU(alpha=1,inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),padding=(1,1)),
            nn.ELU(alpha=1,inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),padding=(1,1)),
            nn.ELU(alpha=1,inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.5)
        )

        self.Linear_layers = nn.Sequential(
            nn.Linear(in_features=int(((img_size/8)**2) *256), out_features=128),
            nn.ELU(alpha=1,inplace=True),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.6),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        
        x = self.Conv_layers(x)
        x = x.view(x.size(0), -1)
        outs = self.Linear_layers(x)

        return outs


if __name__ == '__main__':
    A01 = fer_net(in_chs=3, num_classes = 12, img_size=72)
    print(A01)
    rand_data = torch.rand(12,3,72,72)
    ou1 = A01(rand_data)
    print(ou1.shape)

