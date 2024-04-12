# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import config

n_fft = 512
bins = n_fft // 2 + 1  # Fourier bins


class LogSpectrogram(nn.Module):
    def __init__(self, n_fft=400, hop_length=None, power=1):
        super(LogSpectrogram, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power
        )

    def forward(self, waveform):
        spectrogram = self.spectrogram(waveform)
        return torch.log(1 + spectrogram)


wav2spec = LogSpectrogram(n_fft=n_fft)


def data_processing(data):
    spectrograms = []
    contextual_tokens = []
    len_tokens = []

    for (waveform, chars, phonemes) in data:
        tokens = chars if config.use_chars else phonemes

        # convert waveform to spectrogram
        waveform = torch.Tensor(waveform)
        spec = wav2spec(waveform).unsqueeze(0)  # (channel, bins x time)
        spectrograms.append(spec)

        # extract context for each token
        token_with_context = [tokens[i:i + 2 * config.context + 1] for i in range(len(tokens) - 2 * config.context)]
        contextual_tokens += token_with_context
        len_tokens.append(len(token_with_context))

    spectrograms = torch.stack(spectrograms)
    contextual_tokens = torch.IntTensor(contextual_tokens)

    return spectrograms, contextual_tokens, len_tokens


class RCB(nn.Module):
    """Residual Convolutional Block"""

    def __init__(self, in_channels, out_channels, kernel=3):
        super(RCB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, padding=kernel // 2)
        self.group_norm1 = nn.GroupNorm(num_groups=1 if in_channels == 1 else 4,
                                        num_channels=in_channels)  # how many groups?
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

    def __init__(self, num_RCBs=10):
        super(AudioEncoder, self).__init__()

        self.RCBs = nn.Sequential(
            RCB(1, config.channels),
            *[RCB(config.channels, config.channels) for _ in range(num_RCBs - 1)]
        )

        self.conv1d = nn.Conv1d(in_channels=config.channels, out_channels=config.embedding_dim, kernel_size=bins)

    def forward(self, x):
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
        # (batch, context)
        x = self.embedding_layer(x)  # (batch, context, embedding)

        s = x.shape
        x = x.reshape(s[0], s[1] * s[2])  # (batch, embedding)
        x = self.dense_layers(x)  # (batch, embedding)

        # l2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x


class SimilarityModel(nn.Module):

    def __init__(self, audio_encoder, text_encoder):
        super(SimilarityModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

    def forward(self, spec, pos, len_pos, neg):
        A = self.audio_encoder(spec)
        P = self.text_encoder(pos)
        N = self.text_encoder(neg)

        cumsum = np.cumsum([0] + len_pos)

        PA = torch.empty((len(pos), A.shape[2]))
        NA = torch.empty((len(neg), A.shape[2]))

        for i in range(len(A)):
            j, k = cumsum[i], cumsum[i + 1]
            PA[j:k] = torch.matmul(P[j:k], A[i])  # (samples, time)
            j, k = i * config.num_negative_samples, (i + 1) * config.num_negative_samples
            NA[j:k] = torch.matmul(N[j:k], A[i])  # (samples, time)

        return PA, NA
