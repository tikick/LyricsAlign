# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import torch
import numpy as np
from prettytable import PrettyTable
import config
import torchaudio
import warnings
import librosa
from torch import nn
import string


phone_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
              'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
              'Z', 'ZH', ' ']
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}

#char_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z', "'", ' ']
#char2int = {char_dict[i]: i for i in range(len(char_dict))}


def set_seed(seed=97):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_num_params = 0
    for name, params in model.named_parameters():
        if not params.requires_grad:
            continue
        num_params = params.numel()
        table.add_row([name, num_params])
        total_num_params += num_params
    print(table)
    print(f'Total Trainable Params: {total_num_params}')
    return total_num_params


def encode_chars(words) -> np.ndarray:
    lyrics = ' '.join(words)
    # lyrics = ' '.join(lyrics.split())
    lyrics = ' ' * config.context + lyrics + ' ' * config.context  # padding
    chars = []
    for c in lyrics.lower():
        idx = string.ascii_lowercase.find(c)
        if idx == -1:
            if c == "'":
                idx = 26
            elif c == ' ':
                idx = 27
            else:
                continue  # remove unknown characters
        chars.append(idx)
    return np.array(chars, dtype=np.short)

def encode_phonemes(words) -> np.ndarray:
    phonemes_list = []
    for word_phonemes in words:
        phonemes_list += word_phonemes + [' ']
    phonemes_list = phonemes_list[:-1]
    phonemes_list = [' '] * config.context + phonemes_list + [' '] * config.context  # padding

    phonemes = []
    for p in phonemes_list:
        idx = phone2int[p]
        phonemes.append(idx)

    return np.array(phonemes, dtype=np.short)

def load(path: str, sr: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y, _ = librosa.load(path, sr=sr, res_type='kaiser_fast')

    if len(y.shape) != 1:
        raise ValueError('Waveform has multiple dimensions')

    return y

class LogSpectrogram(nn.Module):
    def __init__(self):
        super(LogSpectrogram, self).__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=config.n_fft, power=1)

    def forward(self, waveform):
        spec = self.spectrogram(waveform)
        return torch.log(1 + spec)
    
def wav2spec(waveform: np.ndarray) -> np.ndarray:
    waveform = torch.Tensor(waveform)
    log_spec = LogSpectrogram()(waveform)
    return log_spec.numpy()
