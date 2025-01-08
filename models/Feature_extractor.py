import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import copy

class Feature_extractor_1DCNN_tiny(nn.Module):
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size = 3, stride = 1, dropout = 0.35):
        super(Feature_extractor_1DCNN_tiny, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x

class Modern(nn.Module):
    def __init__(self, nvars, dmodel, kernel_size = 1, stride = 1, drop = 0.1):
        super(Modern, self).__init__()

        dff = 1 * dmodel

        self.norm = nn.BatchNorm1d(dmodel)
        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 3, 1)
        B, M, D, N = x.shape
        x = x.reshape(B*M,D,N)
        x = self.norm(x)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = x.permute(0, 3, 1, 2)

        x = input + x

        return x