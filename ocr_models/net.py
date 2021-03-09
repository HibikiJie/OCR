from torch import nn
import torch


class BidirectionalLSTM(nn.Module):
    """
    双向LSTM网络层。
    参数：
        input_size：输入特征尺寸
        hidden_size:隐藏层特征尺寸
        output_size：输出特征尺寸
    形状：
        input：（S,N,V）序列、批次、特征尺寸
        output：同输入
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        output = self.fc(recurrent)
        return output


class OcrNet(nn.Module):

    def __init__(self, nclass, imgH=32, nc=3, nh=256, leakyRelu=False):
        super(OcrNet, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, x):
        n, c, h, w = x.shape
        assert h % 16 == 0
        assert w % 4 == 0
        x = self.cnn(x)
        n, c, h, w = x.shape  # n,c,h,w
        assert h == 1
        x = x.reshape(n, c, w)
        x = self.rnn(x.permute(2, 0, 1))
        return x


if __name__ == '__main__':
    m = OcrNet(72)
    x = torch.randn(32, 3, 32, 100)
    print(m(x).shape)
