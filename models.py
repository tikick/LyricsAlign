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
        self.group_norm1 = nn.GroupNorm(num_groups=1 if in_channels == 1 else 4, num_channels=in_channels)  # how many groups?
        self.group_norm2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
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

        assert config.num_RCBs >= 1
        self.RCBs = nn.Sequential(
            RCB(1, config.channels),
            *[RCB(config.channels, config.channels) for _ in range(config.num_RCBs - 1)]
        )

        self.conv1d = nn.Conv1d(in_channels=config.channels, out_channels=config.embedding_dim, kernel_size=config.freq_bins)

    def forward(self, x):
        #print("\tIn AudioEncoder: input size", x.shape)
        # x.shape: (batch, feature, time)

        assert x.shape[1] == config.freq_bins

        x = x.unsqueeze(1)  # (batch, channel, feature, time)

        x = self.RCBs(x)  # (batch, channel, feature, time)

        # 1D conv layer
        # merge batch and time dimension to apply 1D conv to each time bin, then separate the dimensions
        x = x.permute(0, 3, 1, 2)  # (batch, time, channel, feature)
        s = x.shape
        x = x.reshape(s[0] * s[1], s[2], s[3])  # (batch, channel, feature)
        x = self.conv1d(x)
        x = x.reshape(s[0], s[1], config.embedding_dim)  # (batch, time, embedding)
        x = x.transpose(1, 2)  # (batch, embedding, time)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        assert x.shape[1] == config.embedding_dim
        #print("\tIn AudioEncoder: output size", x.shape)
        return x


class TextEncoder(nn.Module):

    def __init__(self):
        super(TextEncoder, self).__init__()

        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.dense_layers = nn.Sequential(
            nn.Linear(config.embedding_dim * (1 + 2 * config.context), config.embedding_dim * (1 + 2 * config.context)),
            nn.ReLU(),
            nn.Linear(config.embedding_dim * (1 + 2 * config.context), config.embedding_dim)
        )

    def forward(self, x):
        #print("\tIn TextEncoder: input size", x.shape)
        # x.shape: (batch, context)

        assert x.shape[1] == 1 + 2 * config.context

        x = self.embedding_layer(x)  # (batch, context, embedding)

        if config.context == 0:
            x = x.squeeze(1)  # (batch, embedding)
        else:
            s = x.shape
            x = x.reshape(s[0], s[1] * s[2])  # (batch, embedding)
            x = self.dense_layers(x)  # (batch, embedding)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        assert x.shape[1] == config.embedding_dim
        #print("\tIn TextEncoder: output size", x.shape)
        return x


class SimilarityModel(nn.Module):

    def __init__(self):
        super(SimilarityModel, self).__init__()
        self.audio_encoder = nn.DataParallel(AudioEncoder())
        self.text_encoder = nn.DataParallel(TextEncoder())

    def forward(self, spectrograms, positives, positives_per_spectrogram=None, negatives=None):
        
        if negatives is not None:  # we're in train
            A = self.audio_encoder(spectrograms)
            P = self.text_encoder(positives)
            N = self.text_encoder(negatives)

            cumsum = np.cumsum([0] + positives_per_spectrogram)

            PA = torch.empty((len(positives), A.shape[2]), device=A.device)  # positive scores
            NA = torch.empty((len(negatives), A.shape[2]), device=A.device)  # negative scores

            for i in range(len(A)):
                j, k = cumsum[i], cumsum[i + 1]
                PA[j:k] = torch.matmul(P[j:k], A[i])  # (tokens, time)
                j, k = i * config.num_negative_samples, (i + 1) * config.num_negative_samples
                NA[j:k] = torch.matmul(N[j:k], A[i])  # (tokens, time)

            return PA, NA
        
        else:  # we're in eval
            assert len(spectrograms) == 1

            A = self.audio_encoder(spectrograms)
            P = self.text_encoder(positives)
            
            S = torch.matmul(P, A[0])
            
            return 0.5 * (S + 1)
    

def _contrastive_loss(PA, NA, times):
    return 2 * (config.alpha * torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
                (1 - config.alpha) * torch.mean(torch.pow(torch.max(NA, dim=1).values, 2)))  # max along time dimension

def contrastive_loss(PA, NA, times):
    assert len(times) == PA.shape[0]
    fps = PA.shape[1] / (config.segment_length / config.sr)
    sum = 0.
    for i, (start, end) in enumerate(times):
        frame_start, frame_end = int(start * fps), int(end * fps)
        assert 0 <= frame_start < PA.shape[1] and 0 <= frame_end < PA.shape[1]
        row_slice = PA[i, :]#frame_start:frame_end + 1]
        sum += torch.pow(torch.max(row_slice) - 1, 2)
    mean_positives = sum / len(times)

    return 2 * (config.alpha * mean_positives + \
                (1 - config.alpha) * torch.mean(torch.pow(torch.max(NA, dim=1).values, 2)))  # max along time dimension
