# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

import config
from media import show_plot


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

    def forward(self, spectrograms, positives, positives_per_spectrogram=None):
        
        if positives_per_spectrogram is not None:  # we're in train
            A = self.audio_encoder(spectrograms)
            P = self.text_encoder(positives)

            cumsum = np.cumsum([0] + positives_per_spectrogram)

            PA = torch.empty((len(positives), A.shape[2]), device=A.device)  # positive scores

            for i in range(len(A)):
                j, k = cumsum[i], cumsum[i + 1]
                PA[j:k] = torch.matmul(P[j:k], A[i])  # (tokens, time)

            return PA
        
        else:  # we're in eval
            assert len(spectrograms) == 1

            A = self.audio_encoder(spectrograms)
            P = self.text_encoder(positives)
            
            S = torch.matmul(P, A[0])
            
            return 0.5 * (S + 1)
    

def _contrastive_loss(PA, NA, times):
    return 2 * (config.alpha * torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
                (1 - config.alpha) * torch.mean(torch.pow(torch.max(NA, dim=1).values, 2)))  # max along time dimension

def contrastive_loss(PA, times, is_duplicate):
    assert len(times) == PA.shape[0] and len(times) == len(is_duplicate)
    duration = config.segment_length / config.sr
    fps = PA.shape[1] / duration
    pos_sum = 0.
    neg_sum = 0.
    neg_summands = 0
    for i, (start, end) in enumerate(times):

        frame_start = int((start - config.box_slack) * fps)
        frame_end = int((end + config.box_slack) * fps)
        assert frame_start <= frame_end
        if config.box_slack > 0:
            frame_start = max(frame_start, 0)
            frame_end = min(frame_end, PA.shape[1] - 1)
        assert 0 <= frame_start <= frame_end < PA.shape[1]

        pos_row_slice = PA[i, frame_start:frame_end + 1]
        pos_sum += torch.pow(torch.max(pos_row_slice) - 1, 2)

        neg_row_slice = torch.cat((PA[i, :frame_start], PA[i, frame_end + 1:]))
        if neg_row_slice.numel() == 0 or is_duplicate[i]:
            continue
        neg_sum += torch.pow(torch.max(neg_row_slice), 2)
        neg_summands += 1

    mean_positives = pos_sum / len(times)
    if neg_summands > 0:
        mean_negatives = neg_sum / neg_summands
    else:
        mean_negatives = 0

    #mean_negatives = torch.mean(torch.pow(torch.max(NA, dim=1).values, 2))  # max along time dimension
    #mean_negatives = torch.mean(torch.pow(NA, 2))

    return 2 * (config.alpha * mean_positives + (1 - config.alpha) * mean_negatives)
