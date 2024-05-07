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
from g2p_en import G2p


phoneme_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
              'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
              'Z', 'ZH', ' ']
phoneme2int = {phoneme_dict[i]: i for i in range(len(phoneme_dict))}

char_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z', "'", ' ']
char2int = {char_dict[i]: i for i in range(len(char_dict))}

g2p = G2p()


def set_seed(seed=97):
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


def words2phowords(words):
    phowords = []
    for word in words:
        word = word.strip("',")  # g2p does not remove leading and trailing ',
        word_phonemes = g2p(word)
        word_phonemes = [p if p[-1] not in string.digits else p[:-1] for p in word_phonemes]
        phowords.append(word_phonemes)
    return phowords

def lines2pholines(lines):
    pass


def encode_words(words, space_padding):
    lyrics = ' '.join(words)
    lyrics = ' ' * space_padding + lyrics + ' ' * space_padding
    
    chars = []
    for c in lyrics.lower():
        try:
            idx = char2int[c]
        except KeyError:
            pass  # remove unknown characters
        chars.append(idx)
    return chars

def encode_phowords(phowords, space_padding):
    phonemes_list = []
    for word_phonemes in phowords:
        phonemes_list += word_phonemes + [' ']
    phonemes_list = phonemes_list[:-1]
    phonemes_list = [' '] * space_padding + phonemes_list + [' '] * space_padding

    phonemes = []
    for p in phonemes_list:
        idx = phoneme2int[p]
        phonemes.append(idx)
    return phonemes


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
