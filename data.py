# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import DALI as dali_code
import bisect
import numpy as np
import os
from tqdm import tqdm
import h5py
import pickle
from math import floor
import torch
from torch.utils.data import Dataset

import config
from utils import encode_chars, encode_phonemes, load, wav2spec

phone_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
              'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
              'Z', 'ZH', ' ']
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}

#char_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z', "'", ' ']
#char2int = {char_dict[i]: i for i in range(len(char_dict))}


def get_dali(lang='english'):
    dali_data = dali_code.get_the_DALI_dataset(config.dali_annot_path, skip=[],
        keep=[])
        #keep=['0a3cd469757e470389178d44808273ab', '0a81772ae3a7404f9ef09ecd1f94db07', '0dea06fa7ca04eb88b17e8d83993adc3', '1ae34dc139ea43669501fb9cef85cbd0', '1afbb77f88dc44e9bedc07b54341be9c', '1b9c139f491c41f5b0776eefd21c122d'])

    lang_subset = []

    audio_files = os.listdir(config.dali_audio_path)
    for file in audio_files:
        if os.path.exists(os.path.join(config.dali_annot_path, file[:-4] + '.gz')):
            # get annotations and metadata for the current song
            annot = dali_data[file[:-4]].annotations['annot']
            metadata = dali_data[file[:-4]].info['metadata']

            if lang is not None and metadata['language'] != lang:
                continue

            song = {'id': file[:-4],
                    'audio_path': os.path.join(config.dali_audio_path, file),
                    'words': annot['words'],
                    'phonemes': annot['phonemes']
                    }

            lang_subset.append(song)

    return lang_subset


class DaliDatasetHDF5(Dataset):

    def __init__(self, dataset, partition, in_memory=False):
        super(DaliDatasetHDF5, self).__init__()

        hdf_file_path = os.path.join(config.hdf_dir, partition + '.hdf5')

        if not os.path.exists(hdf_file_path):
            if not os.path.exists(config.hdf_dir):
                os.makedirs(config.hdf_dir)

            print(f'Creating {partition} samples')
            sample_id = 0
            
            with h5py.File(hdf_file_path, 'w', libver='latest') as f:

                for song in tqdm(dataset):
                    waveform = load(song['audio_path'], sr=config.sr)

                    start_times = [d['time'][0] for d in song['words']]
                    end_times = [d['time'][1] for d in song['words']]

                    max_num_samples = floor((len(waveform) - config.sample_length) / config.hop_size) + 1
                    for i in range(max_num_samples):
                        sample_start = i * config.hop_size
                        sample_end = sample_start + config.sample_length
                        assert sample_end <= len(waveform)

                        # find the lyrics within (start, end)
                        idx_first_word = bisect.bisect_left(start_times, sample_start / config.sr)
                        idx_last_word = bisect.bisect_right(end_times, sample_end / config.sr) - 1

                        if idx_first_word > idx_last_word:  # no words (fully contained) in this sample, skip
                            continue

                        # convert characters/phonemes present in (start, end) to numerical representation
                        chars = encode_chars(song['words'][idx_first_word:idx_last_word + 1])
                        phonemes = encode_phonemes(song['phonemes'][idx_first_word:idx_last_word + 1])

                        spec = wav2spec(waveform[sample_start:sample_end])

                        # sample = (spec, chars, phonemes)
                        grp = f.create_group(str(sample_id))
                        grp.create_dataset('spectrogram', data=spec)
                        grp.create_dataset('chars', data=chars)
                        grp.create_dataset('phonemes', data=phonemes)
                        sample_id += 1
                
                f.attrs['num_samples'] = sample_id

        driver = 'core' if in_memory else None  # Load hdf file fully into memory if desired
        self.hdf_file = h5py.File(hdf_file_path, 'r', driver=driver, libver='latest')
        self.length = self.hdf_file.attrs['num_samples']

    def __getitem__(self, index):
        spec = self.hdf_file[str(index)]['spectrogram']
        chars = self.hdf_file[str(index)]['chars']
        phonemes = self.hdf_file[str(index)]['phonemes']
        return spec, chars, phonemes

    def __len__(self):
        return self.length

    def close(self):
        self.hdf_file.close()


class DaliDatasetPickle(Dataset):
    def __init__(self, dataset, partition):
        super(DaliDatasetPickle, self).__init__()

        pickle_file = os.path.join(config.pickle_dir, partition + '.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print(f'Creating {partition} samples')
            samples = []
            for song in tqdm(dataset):
                waveform = load(song['audio_path'], sr=config.sr)

                start_times = [d['time'][0] for d in song['words']]
                end_times = [d['time'][1] for d in song['words']]

                max_num_samples = floor(((len(waveform) - config.sample_length) / config.hop_size) + 1)
                for i in range(max_num_samples):
                    sample_start = i * config.hop_size
                    sample_end = sample_start + config.sample_length
                    assert sample_end <= len(waveform)

                    # find the lyrics within (start, end)
                    idx_first_word = bisect.bisect_left(start_times, sample_start / config.sr)
                    idx_last_word = bisect.bisect_right(end_times, sample_end / config.sr) - 1

                    if idx_first_word > idx_last_word:  # no words (fully contained) in this sample, skip
                        continue

                    # convert characters/phonemes present in (start, end) to numerical representation
                    chars = encode_chars(song['words'][idx_first_word:idx_last_word + 1])
                    phonemes = encode_phonemes(song['phonemes'][idx_first_word:idx_last_word + 1])

                    spec = wav2spec(waveform[sample_start:sample_end])
                    
                    sample = (spec, chars, phonemes)
                    samples.append(sample)

            # write samples onto pickle file
            with open(pickle_file, 'wb') as f:
                pickle.dump(samples, f)

        # load samples from pickle file
        with open(pickle_file, 'rb') as f:
            self.samples = pickle.load(f)

    def __getitem__(self, index):
        return self.samples[index]  # (spec, chars, phonemes)

    def __len__(self):
        return len(self.samples)


class LyricsDatabase:
    def __init__(self, dataset):

        pickle_file = os.path.join(config.pickle_dir, 'neg_probs.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print('Computing negative sampling probabilities')

            assert config.context <= 1
            frequencies = np.zeros((pow(config.vocab_size, 2 * config.context + 1),), dtype=int)

            for song in tqdm(dataset):
                if config.use_chars:
                    tokens = encode_chars(song['words'])
                else:
                    tokens = encode_phonemes(song['phonemes'])

                token_with_context = [tokens[i:i + 2 * config.context + 1] for i in range(len(tokens) - 2 * config.context)]
                for contextual_token in token_with_context:
                    idx = self._contextual_token2idx(contextual_token)
                    frequencies[idx] += 1
            
            # write frequencies onto pickle file
            with open(pickle_file, 'wb') as f:
                pickle.dump(frequencies, f)

        # load frequencies from pickle file
        with open(pickle_file, 'rb') as f:
            self.frequencies = pickle.load(f)

    def sample(self, num_samples, pos, len_pos):
        # to avoid sampling positives, set frequency of positives to 0, sample negatives, and restore the original frequencies

        contextual_tokens = []
        cumsum = np.cumsum([0] + len_pos)

        for i in range(len(len_pos)):
            j, k = cumsum[i], cumsum[i + 1]

            # set frequencies of positive samples to 0
            original_freq = []
            for l in range(j, k):
                contextual_token = pos[j + l]
                idx = self._contextual_token2idx(contextual_token)
                original_freq.append(self.frequencies[idx])
                self.frequencies[idx] = 0
            
            # sample negatives
            prob = self.frequencies / np.sum(self.frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            contextual_tokens += [self._idx2contextual_token(idx) for idx in indices]

            # restore original frequencies
            for l in range(j, k):
                contextual_token = pos[j + l]
                idx = self._contextual_token2idx(contextual_token)
                self.frequencies[idx] = original_freq[l]

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


def collate(data):
    spectrograms = []
    contextual_tokens = []
    len_tokens = []

    for spec, chars, phonemes in data:
        tokens = chars if config.use_chars else phonemes

        spectrograms.append(spec)

        # extract context for each token
        token_with_context = [tokens[i:i + 2 * config.context + 1] for i in range(len(tokens) - 2 * config.context)]
        contextual_tokens += token_with_context
        len_tokens.append(len(token_with_context))

    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    spectrograms = torch.Tensor(np.array(spectrograms))
    contextual_tokens = torch.IntTensor(np.array(contextual_tokens))

    return spectrograms, contextual_tokens, len_tokens