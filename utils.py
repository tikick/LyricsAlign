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
import csv
import os


phoneme_dict = [' ', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
phoneme2int = {phoneme_dict[i]: i for i in range(len(phoneme_dict))}
int2phoneme = {i: phoneme_dict[i] for i in range(len(phoneme_dict))}

char_dict = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
char2int = {char_dict[i]: i for i in range(len(char_dict))}
int2char = {i: char_dict[i] for i in range(len(char_dict))}

g2p = G2p()


def fix_seed(seed=97):
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
        raw_word = word
        word = word.strip("'")  # g2p does not remove leading and trailing '
        phoword = g2p(word)
        phoword = [p[:-1] if p[-1] in string.digits else p for p in phoword]
        phowords.append(phoword)

        for p in phoword:
            if p not in phoneme_dict[1:]:
                raise NotImplemented(f'Unknown phoneme "{p}" in word "{raw_word}"')

    return phowords

def lines2pholines(lines):
    pholines = []
    for line in lines:
        words = line.split()
        phowords = words2phowords(words)
        pholine = []
        for phoword in phowords:
            pholine += phoword + [' ']
        pholine = pholine[:-1]  # remove last space
        pholines.append(pholine)
    return pholines


def normalize_dali_annot(raw_words, raw_times, cut=False):
    # if cut=True removes the whole word, else strips the unknown chars from the word
    # stripping away punctuation and only then cutting might be better
    words = []
    times = []
    unk_chars = 0
    total_chars = 0
    for raw_word, raw_time in zip(raw_words, raw_times):
        total_chars += len(raw_word)
        word = ''.join(filter(lambda c: c in char_dict[1:], raw_word.lower()))
        if len(word) == 0 or \
           len(word) < len(raw_word) and (cut or len(word) > 15):  # len(word) > 15: raw_word is likely multiple words separated by special chars, e.g. -
            continue
        words.append(word)
        times.append(raw_time)
        unk_chars += len(raw_word) - len(word)
    return words, times, unk_chars, total_chars


def encode_chars(chars):
    enc_chars = []
    for c in chars:
        idx = char2int[c]
        enc_chars.append(idx)
    return enc_chars

def encode_phonemes(phonemes):
    enc_phonemes = []
    for p in phonemes:
        idx = phoneme2int[p]
        enc_phonemes.append(idx)
    return enc_phonemes

def encode_words(words, space_padding):
    lyrics = ' '.join(words)
    lyrics = ' ' * space_padding + lyrics + ' ' * space_padding
    
    enc_chars = []
    for c in lyrics:
        idx = char2int[c]
        enc_chars.append(idx)
    return enc_chars

def encode_phowords(phowords, space_padding):
    phonemes = []
    for phoword in phowords:
        phonemes += phoword + [' ']
    phonemes = phonemes[:-1]
    phonemes = [' '] * space_padding + phonemes + [' '] * space_padding

    enc_phonemes = []
    for p in phonemes:
        idx = phoneme2int[p]
        enc_phonemes.append(idx)
    return enc_phonemes


def read_gt_alignment(gt_file):
    gt_alignment = []
    with open(gt_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            gt_alignment.append((float(row['word_start']), float(row['word_end'])))
    return gt_alignment


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




def gen_jamendo_segment():
    from utils import load, read_gt_alignment
    import config
    import soundfile as sf
    import os
    import csv

    sr = 44100
    audio_file = 'Color_Out_-_Falling_Star.mp3'

    start = 59
    end = 82

    audio = load(os.path.join(config.jamendo_audio, audio_file), sr=sr)
    audio_segment = audio[start * sr:end * sr]
    audio_segment_file = str(start) + '-' + str(end) + '_' + audio_file
    sf.write(os.path.join(config.jamendo_segments_audio, audio_segment_file), audio_segment, sr)

    with open(os.path.join(config.jamendo_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if len(l) > 0]  # remove empty lines between paragraphs
    words = ' '.join(lines).split()
    times = read_gt_alignment(os.path.join(config.jamendo_annotations, audio_file[:-4] + '.csv'))
    segment_words = []
    segment_times = []
    for word, time in zip(words, times):
        if time[0] >= start and time[1] <= end:
            segment_words.append(word)
            segment_times.append((time[0] - start, time[1] - start))

    with open(os.path.join(config.jamendo_segments_lyrics, audio_segment_file[:-4] + '.txt'), 'w') as f:
        f.write(' '.join(segment_words))

    header = ['word_start', 'word_end']
    with open(os.path.join(config.jamendo_segments_annotations, audio_segment_file[:-4] + '.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(segment_times)

    with open(config.jamendo_segments_metadata, 'a') as f:
        f.write(f'\n_,{audio_segment_file},_,_,_,_,English,_,_,_')


if __name__ == '__main__':
    print('Running utils.py')
    gen_jamendo_segment()