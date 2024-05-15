# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class RCB(nn.Module):
    """Residual Convolutional Block"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RCB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm1 = nn.GroupNorm(num_groups=1 if in_channels == 1 else 4,
                                        num_channels=in_channels)  # how many groups?
        self.group_norm2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x):
        residual = x    # (batch, channel, feature, time)
        x = self.group_norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.group_norm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.RCBs = nn.Sequential(
            RCB(1, config.channels),
            *[RCB(config.channels, config.channels) for _ in range(config.num_RCBs - 1)]
        )

        self.conv1d = nn.Conv1d(in_channels=config.channels, out_channels=config.embedding_dim, kernel_size=config.fourier_bins)

    def forward(self, x):
        # x.shape: (batch, feature, time)
        x = x.unsqueeze(1)  # (batch, channel, feature, time)

        x = self.RCBs(x)  # (batch, channel, feature, time)

        # 1D conv layer
        # Merge batch and time dimension to apply 1D conv to each time bin, then separate the dimensions
        x = x.permute(0, 3, 1, 2)  # (batch, time, channel, feature)
        s = x.shape
        x = x.reshape(s[0] * s[1], s[2], s[3])  # (batch, channel, feature)
        x = self.conv1d(x)
        x = x.reshape(s[0], s[1], config.embedding_dim)  # (batch, time, embedding)
        x = x.transpose(1, 2)  # (batch, embedding, time)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x


class TextEncoder(nn.Module):

    def __init__(self):
        super(TextEncoder, self).__init__()

        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.dense_layers = nn.Sequential(
            nn.Linear(config.embedding_dim * (2 * config.context + 1), config.embedding_dim * (2 * config.context + 1)),
            nn.ReLU(),
            nn.Linear(config.embedding_dim * (2 * config.context + 1), config.embedding_dim)
        )

    def forward(self, x):
        assert x.shape[1] == 1 + 2 * config.context

        # (batch, context)
        x = self.embedding_layer(x)  # (batch, context, embedding)

        s = x.shape
        x = x.reshape(s[0], s[1] * s[2])  # (batch, embedding)
        x = self.dense_layers(x)  # (batch, embedding)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x


class SimilarityModel(nn.Module):

    def __init__(self):
        super(SimilarityModel, self).__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, spec, pos, len_pos=None, neg=None):
        # if len_pos=None, neg=None then we're in eval, otherwise in train
        
        if neg is not None:
            A = self.audio_encoder(spec)
            P = self.text_encoder(pos)
            N = self.text_encoder(neg)

            cumsum = np.cumsum([0] + len_pos)

            PA = torch.empty((len(pos), A.shape[2]), device=A.device)
            NA = torch.empty((len(neg), A.shape[2]), device=A.device)

            for i in range(len(A)):
                j, k = cumsum[i], cumsum[i + 1]
                PA[j:k] = torch.matmul(P[j:k], A[i])  # (samples, time)
                j, k = i * config.num_negative_samples, (i + 1) * config.num_negative_samples
                NA[j:k] = torch.matmul(N[j:k], A[i])  # (samples, time)

            return PA, NA
        
        else:
            assert spec.shape[0] == 1

            A = self.audio_encoder(spec)
            P = self.text_encoder(pos)
            
            S = torch.matmul(P, A[0])
            
            return 0.5 * (S + 1)
    

#def contrastive_loss(PA, NA):
#    return torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
#           torch.mean(torch.pow(torch.max(NA, dim=1).values, 2))  # max along time dimension

def contrastive_loss(PA, NA):
    PA_max = torch.pow(torch.max(PA, dim=1).values - 1, 2)
    NA_max = torch.pow(torch.max(NA, dim=1).values, 2)
    return (torch.sum(PA_max) + torch.sum(NA_max)) / (len(PA_max) + len(NA_max))




class TimeAudioEncoder(nn.Module):

    def __init__(self):
        super(TimeAudioEncoder, self).__init__()

        self.RCBs = nn.Sequential(
            RCB(1, config.channels),
            *[RCB(config.channels, config.channels) for _ in range(config.num_RCBs - 1)]
        )

        self.conv1d = nn.Conv1d(in_channels=config.channels, out_channels=config.embedding_dim, kernel_size=config.fourier_bins)

    def forward(self, x):
        # x.shape: (batch, feature, time)
        x = x.unsqueeze(1)  # (batch, channel, feature, time)

        config.time_report.start_timer('audio_encoder.RCBs')
        x = self.RCBs(x)  # (batch, channel, feature, time)
        torch.cuda.synchronize()
        config.time_report.end_timer('audio_encoder.RCBs')

        # 1D conv layer
        # Merge batch and time dimension to apply 1D conv to each time bin, then separate the dimensions
        x = x.permute(0, 3, 1, 2)  # (batch, time, channel, feature)
        s = x.shape
        x = x.reshape(s[0] * s[1], s[2], s[3])  # (batch, channel, feature)

        config.time_report.start_timer('audio_encoder.conv1d')
        x = self.conv1d(x)
        torch.cuda.synchronize()
        config.time_report.end_timer('audio_encoder.conv1d')

        x = x.reshape(s[0], s[1], config.embedding_dim)  # (batch, time, embedding)
        x = x.transpose(1, 2)  # (batch, embedding, time)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x
    

class TimeSimilarityModel(nn.Module):

    def __init__(self):
        super(TimeSimilarityModel, self).__init__()
        self.audio_encoder = TimeAudioEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, spec, pos, len_pos, neg):

        config.time_report.start_timer('audio_encoder')
        A = self.audio_encoder(spec)
        torch.cuda.synchronize()
        config.time_report.end_timer('audio_encoder')

        config.time_report.start_timer('text_encoder')
        P = self.text_encoder(pos)
        N = self.text_encoder(neg)
        torch.cuda.synchronize()
        config.time_report.end_timer('text_encoder')

        cumsum = np.cumsum([0] + len_pos)

        config.time_report.start_timer('similarity')
        PA = torch.empty((len(pos), A.shape[2]), device=A.device)
        NA = torch.empty((len(neg), A.shape[2]), device=A.device)
        for i in range(len(A)):
            j, k = cumsum[i], cumsum[i + 1]
            PA[j:k] = torch.matmul(P[j:k], A[i])  # (samples, time)
            j, k = i * config.num_negative_samples, (i + 1) * config.num_negative_samples
            NA[j:k] = torch.matmul(N[j:k], A[i])  # (samples, time)
        torch.cuda.synchronize()
        config.time_report.end_timer('similarity')

        return PA, NA
