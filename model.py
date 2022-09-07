import torch
from torch import flatten
import torch.nn as nn

class DCNN(nn.Module):
    def __init__(self, args):
        super(DCNN, self).__init__()
        self.args = args

        self.sq1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(3, 2)
        self.bn1 = nn.BatchNorm2d(96)
        self.pad1 = nn.ZeroPad2d(2)

        self.sq2_1 = nn.Sequential(
            nn.Conv2d(48, 128, 5),
            nn.ReLU(),
        )
        self.sq2_2 = nn.Sequential(
            nn.Conv2d(48, 128, 5),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(3, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pad2 = nn.ZeroPad2d(1)


        self.sq3 = nn.Sequential(
            nn.Conv2d(256, 384, 3),
            nn.ReLU(),
        )
        self.pad3 = nn.ZeroPad2d(1)

        
        self.sq4_1 = nn.Sequential(
            nn.Conv2d(192, 192, 3),
            nn.ReLU(),
        )
        self.sq4_2 = nn.Sequential(
            nn.Conv2d(192, 192, 3),
            nn.ReLU(),
        )
        self.pad4 = nn.ZeroPad2d(1)


        self.sq5_1 = nn.Sequential(
            nn.Conv2d(192, 128, 3),
            nn.ReLU(),
        )
        self.sq5_2 = nn.Sequential(
            nn.Conv2d(192, 128, 3),
            nn.ReLU(),
        )
        self.pool5 = nn.MaxPool2d(3, 2)
        self.bn5 = nn.BatchNorm2d(256)

        self.sq6 = nn.Sequential(
            nn.Linear(in_features = 9216, out_features = 4096),
            nn.ReLU(),
            nn.Dropout2d(p=self.args.dropout)
        )

        self.sq7 = nn.Sequential(
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(),
            nn.Dropout2d(p=self.args.dropout)
        )

        self.sq8 = nn.Sequential(
            nn.Linear(in_features = 4096, out_features = 10),
            nn.ReLU()
        )

        self.ac1 = nn.Softmax()

    def forward(self, x):
        # print(f'Before sq1: {x.shape}')
        x = self.sq1(x)
        # print(f'After sq1: {x.shape}')
        x = self.pool1(x)
        # print(f'After pool1: {x.shape}')
        x = self.bn1(x)
        x = self.pad1(x)
        # print(f'After pad1: {x.shape}')

        
        
        x_1 = self.sq2_1(x[:, 0:48])
        # print(f'After sq2_1: {x_1.shape}')
        x_2 = self.sq2_2(x[:, 48:])
        # print(f'After sq2_2: {x_2.shape}')
        x = torch.cat((x_1, x_2), dim=1)
        # print(f'After concat x: {x.shape}')
        
        x = self.pool2(x)
        # print(f'After pool2: {x.shape}')
        x = self.bn2(x)
        x = self.pad2(x)
        # print(f'After pad2: {x.shape}')

        

        x = self.sq3(x)
        # print(f'After sq3: {x.shape}')
        x = self.pad3(x)
        # print(f'After pad3: {x.shape}')

        
        
        x_4_1 = self.sq4_1(x[:, 0:192])
        # print(f'After sq4_1: {x_1.shape}')
        x_4_2 = self.sq4_2(x[:, 192:])
        # print(f'After sq4_2: {x_2.shape}')
        x = torch.cat((x_4_1, x_4_2), dim=1)
        # print(f'After concat x: {x.shape}')
        
        x = self.pad4(x)
        # print(f'After pad4: {x.shape}')

        
        

        
        x_5_1 = self.sq5_1(x[:, 0:192])
        # print(f'After sq5_1: {x_1.shape}')
        x_5_2 = self.sq5_2(x[:, 192:])
        # print(f'After sq5_2: {x_2.shape}')
        x = torch.cat((x_5_1, x_5_2), dim=1)
        # print(f'After concat x: {x.shape}')
        
        x = self.pool5(x)
        # print(f'After pool5: {x.shape}')
        x = self.bn5(x)

        x = flatten(x, start_dim = 1)
        # print(f'After flatten: {x.shape}')
        x = self.sq6(x)
        # print(f'After sq6: {x.shape}')
        x = self.sq7(x)
        # print(f'After sq7: {x.shape}')
        x = self.sq8(x)
        # print(f'After sq8: {x.shape}')
        x = self.ac1(x)
        return x
        
