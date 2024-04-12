# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import DALI as dali_code
import bisect
import numpy as np
import os
import pickle
import string
from math import floor
from torch.utils.data import Dataset

import config
from utils import load, load_lyrics, gen_phone_gt

phone_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
              'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
              'Z', 'ZH', ' ']
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}
char_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z', "'", ' ']
char2int = {char_dict[i]: i for i in range(len(char_dict))}


def encode_chars(dict_list):
    lyrics_list = [d['text'] for d in dict_list]
    lyrics = ' '.join(lyrics_list)
    # lyrics = ' '.join(lyrics.split())
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
    return chars


def encode_phonemes(dict_list):
    phonemes_nested_list = [d['text'] for d in dict_list]

    phonemes_list = []
    for i, word_phonemes in enumerate(phonemes_nested_list):
        phonemes_list.extend(word_phonemes)
        if i < len(phonemes_nested_list) - 1:
            phonemes_list.append(' ')

    phonemes = []
    for p in phonemes_list:
        idx = phone2int[p]
        phonemes.append(idx)

    return phonemes


def get_dali(lang='english'):
    dali_data = dali_code.get_the_DALI_dataset(config.dali_annot_path, skip=[], keep=[])

    lang_subset = []

    audio_files = os.listdir(config.dali_audio_path)
    for file in audio_files:
        if os.path.exists(os.path.join(config.dali_annot_path, file[:-4] + '.gz')):
            # get annotations and info for the current song
            annot = dali_data[file[:-4]].annotations['annot']
            info = dali_data[file[:-4]].info

            # language filter
            if lang is not None and info['metadata']['language'] != lang:
                continue

            song = {'id': file[:-4],
                    'path': os.path.join(config.dali_audio_path, file),
                    'words': annot['words'],
                    'phonemes': annot['phonemes']
                    }

            lang_subset.append(song)

    return lang_subset


class LyricsAlignDataset(Dataset):
    def __init__(self, dataset, partition):
        super(LyricsAlignDataset, self).__init__()

        pickle_file = os.path.join(config.pickle_dir, partition + '_' + str(config.sr) + '.pkl')

        # Check if pickle file needs to be written
        if not os.path.exists(pickle_file):
            # Create pickle folder if it doesn't exist
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print('Creating {partition} samples')
            samples = []
            for song in dataset:
                waveform = load(song['path'], sr=config.sr)

                start_times = [d['time'][0] for d in song['words']]
                end_times = [d['time'][1] for d in song['words']]

                max_num_samples = floor(((len(waveform) - config.input_length) / config.hop_size) + 1)
                for i in range(max_num_samples):
                    sample_start = i * config.hop_size
                    sample_end = sample_start + config.input_length
                    assert sample_end <= len(waveform)

                    # find the lyrics within (start, end)
                    idx_first_word = bisect.bisect_left(start_times,
                                                        sample_start / config.sr)  # could use increasing iterator to
                    # avoid log factor
                    idx_last_word = bisect.bisect_right(end_times, sample_end / config.sr) - 1

                    if idx_first_word > idx_last_word:  # invalid sample, skip
                        continue

                    # convert characters/phonemes present in (start, end) to numerical representation
                    chars = encode_chars(song['words'][idx_first_word:idx_last_word + 1])
                    phonemes = encode_phonemes(song['phonemes'][idx_first_word:idx_last_word + 1])

                    sample = (waveform[sample_start:sample_end], chars, phonemes)
                    samples.append(sample)

            # write samples onto pickle file
            with open(pickle_file, 'wb') as f:
                pickle.dump(samples, f)

        # load samples from pickle file
        with open(pickle_file, 'rb') as f:
            self.samples = pickle.load(f)

    def __getitem__(self, index):
        return self.samples[index]  # (waveform, chars, phonemes)

    def __len__(self):
        return len(self.samples)


class LyricsDatabase:
    def __init__(self, dataset):

        frequencies = np.zeros((pow(config.vocab_size, 2 * config.context + 1),), dtype=int)

        for song in dataset:
            if config.use_chars:
                tokens = encode_chars(song['words'])
            else:
                tokens = encode_phonemes(song['phonemes'])

            token_with_context = [tokens[i:i + 2 * config.context + 1] for i in range(len(tokens) - 2 * config.context)]
            for contextual_token in token_with_context:
                idx = self._contextual_token2idx(contextual_token)
                frequencies[idx] += 1

        self.prob = frequencies / np.sum(frequencies)

    def sample(self, num_samples):
        indices = np.random.choice(len(self.prob), size=num_samples, p=self.prob)
        contextual_tokens = [self._idx2contextual_token(idx) for idx in indices]
        return contextual_tokens

    @staticmethod
    def _contextual_token2idx(contextual_token):
        idx = 0
        for t in contextual_token:
            idx *= config.vocab_size
            idx += t
        return idx

    @staticmethod
    def _idx2contextual_token(idx):
        contextual_token = []
        for _ in range(1 + 2 * config.context):
            contextual_token.append(idx % config.vocab_size)
            idx = idx // config.vocab_size
        contextual_token = list(reversed(contextual_token))
        return contextual_token
