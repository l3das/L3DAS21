import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from utility_tac.models import *
from torchvision import models
import utility_functions as uf

'''
Pytorch implementation of SELDNet: https://arxiv.org/pdf/1807.00129.pdf
'''

class Fake_Seldnet(nn.Module):
    def __init__(self, dropout_perc=0.5):
        super(Fake_Seldnet, self).__init__()
        self.feat_extraction = models.vgg16.feature()

        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1))

                                #change num output classes
        model.classifier[6] =nn.Linear(in_features=4096,
                                out_features=p['output_classes'], bias=True)

    def forward(x,self):
        x = self.features(x)
        return x


class Seldnet_vanilla(nn.Module):
    def __init__(self, time_dim, freq_dim=256, output_classes=14,
                 pool_size=[[8,2],[8,2],[2,2]], pool_time=False,
                 rnn_size=128, n_rnn=2,
                 fc_size=128, dropout_perc=0., n_cnn_filters=64,
                 verbose=True):
        super(Seldnet_vanilla, self).__init__()
        self.verbose = verbose
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        doa_output_size = output_classes * 3    #here 3 is the max number of simultaneus sounds from the same class
        sed_output_size = doa_output_size * 3   #here 3 is the number of spatial dimensions xyz
        if pool_time:
            self.time_pooled_size = int(time_dim / np.prod(np.array(pool_size), axis=0)[-1])
        else:
            self.time_pooled_size = time_dim
        #building CNN feature extractor
        conv_layers = []
        in_chans = 4
        for p in pool_size:
            curr_chans = n_cnn_filters
            if pool_time:
                pool = [p[0],p[1]]
            else:
                pool = [p[0],1]
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans,
                                kernel_size=3, stride=1, padding=1),  #padding 1 = same with kernel = 3
                    nn.BatchNorm2d(n_cnn_filters),
                    nn.ReLU(),
                    nn.MaxPool2d(pool),
                    nn.Dropout(dropout_perc)))
            in_chans = curr_chans

        self.cnn = nn.Sequential(*conv_layers)

        self.rnn = nn.GRU(128, rnn_size, num_layers=n_rnn, batch_first=True,
                          bidirectional=True, dropout=dropout_perc)

        self.sed = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, doa_output_size),
                    nn.Tanh())

        self.doa = nn.Sequential(
                    nn.Linear(256, fc_size),
                    nn.Dropout(dropout_perc),
                    nn.Linear(fc_size, sed_output_size),
                    nn.Sigmoid())

    def forward(self, x):
        x = self.cnn(x)
        if self.verbose:
            print ('cnn out ', x.shape)    #target dim: [batch, n_cnn_filters, 2, time_frames]
        x = x.permute(0,3,1,2) #[batch, time, channels, freq]
        if self.verbose:
            print ('permuted: ', x.shape)    #target dim: [batch, time_frames, n_cnn_filters, 2]
        x = x.reshape(x.shape[0], self.time_pooled_size, -1)
        if self.verbose:
            print ('reshaped: ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        x, h = self.rnn(x)
        if self.verbose:
            print ('rnn out:  ', x.shape)    #target dim: [batch, 2*n_cnn_filters]
        sed = self.sed(x)
        doa = self.doa(x)
        if self.verbose:
            print ('sed prediction:  ', sed.shape)  #target dim: [batch, time, sed_output_size]
            print ('doa prediction: ', doa.shape)  #target dim: [batch, time, doa_output_size]

        return x

def test_model():
    sample = np.ones((4,32000*60))

    nperseg = 512
    noverlap = 112

    sp = uf.spectrum_fast(sample, nperseg=nperseg, noverlap=noverlap, output_phase=False)
    #sp = np.reshape(sp, (sp.shape[1], sp.shape[0], sp.shape[2]))
    sp = torch.tensor(sp.reshape(1,sp.shape[0],sp.shape[1],sp.shape[2])).float()

    time_dim = 512
    freq_dim = 256
    x = torch.rand(1, 4, freq_dim, time_dim)
    print ('In shapes ', sp.shape, x.shape)
    print (sp.shape[-1])
    print
    model = Seldnet_vanilla(time_dim, pool_time=False)
    model2 = Seldnet_vanilla(sp.shape[-1],pool_time=True)
    #print (model)
    x1 = model(x)
    print ('SP')
    x1 = model2(sp)

    print (x1.shape)

if __name__ == '__main__':
    test_model()
