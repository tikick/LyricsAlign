# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import DALI as dali_code
import bisect
import os
from tqdm import tqdm
import pickle
import csv
from math import floor
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config
from utils import encode_words, encode_phowords, words2phowords, lines2pholines, \
    load, wav2spec, read_jamendo_times, normalize_dali, normalize_jamendo


def get_dali(lang='english'):
    # 96569 of 5069058 chars in DALI are not in utils.char_dict and thus removed in normalize_dali (2% noise)

    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[],
        keep=[])
        #keep=['0a3cd469757e470389178d44808273ab', '0a81772ae3a7404f9ef09ecd1f94db07', '0dea06fa7ca04eb88b17e8d83993adc3', '1ae34dc139ea43669501fb9cef85cbd0', '1afbb77f88dc44e9bedc07b54341be9c', '1b9c139f491c41f5b0776eefd21c122d'])

    songs = []

    audio_files = os.listdir(config.dali_audio)  # only get songs for which we have audio files
    for file in audio_files:
        annot = dali_data[file[:-4]].annotations['annot']
        metadata = dali_data[file[:-4]].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue
        
        words = [d['text'] for d in annot['words']]
        times = [d['time'] for d in annot['words']]
        words, times = normalize_dali(words, times)
        phowords = words2phowords(words)  #[d['text'] for d in annot['phonemes']]

        song = {'id': file[:-4],
                'audio_path': os.path.join(config.dali_audio, file),
                'words': words,
                'phowords': phowords,
                'times': times}

        songs.append(song)

    return songs


def get_jamendo(lang='English'):
    songs = []

    with open(config.jamendo_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendo_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = normalize_jamendo(lines)
            words = ' '.join(lines).split()
            phowords = words2phowords(words)
            pholines = lines2pholines(lines)
            times = read_jamendo_times(os.path.join(config.jamendo_annotations, audio_file[:-4] + '.csv'))
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendo_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': times}
            
            songs.append(song)
    
    return songs


def get_jamendoshorts(lang='English'):
    songs = []

    with open(config.jamendoshorts_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendoshorts_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = normalize_jamendo(lines)
            words = ' '.join(lines).split()
            phowords = words2phowords(words)
            pholines = lines2pholines(lines)
            times = read_jamendo_times(os.path.join(config.jamendoshorts_annotations, audio_file[:-4] + '.csv'))
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendoshorts_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': times
                    }
            
            songs.append(song)
    
    return songs


def jamendo_collate(song):
    waveform = load(song['audio_path'], sr=config.sr)
    spec = wav2spec(waveform)
    spectrogram, all_tokens, _ = collate(data=[(spec, song['words'], song['phowords'])])
    return spectrogram, all_tokens


def collate(data):
    spectrograms = []
    all_tokens = []
    tokens_per_spectrogram = []

    for spec, words, phowords in data:
        spectrograms.append(spec)

        if config.use_chars:
            tokens = encode_words(words)
        else:
            tokens = encode_phowords(phowords)

        tokens_per_spectrogram.append(len(tokens))
        all_tokens += tokens

    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    spectrograms = torch.Tensor(np.array(spectrograms))
    all_tokens = torch.IntTensor(all_tokens)

    return spectrograms, all_tokens, tokens_per_spectrogram


class DaliDataset(Dataset):
    def __init__(self, dataset, partition):
        super(DaliDataset, self).__init__()

        pickle_file = os.path.join(config.pickle_dir, 'dali_' + partition + '.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print(f'Creating {partition} samples')
            samples = []
            for song in tqdm(dataset):
                waveform = load(song['audio_path'], sr=config.sr)

                start_times = [start_time for (start_time, _) in song['times']]
                end_times = [end_time for (_, end_time) in song['times']]

                max_num_samples = floor(((len(waveform) - config.segment_length) / config.hop_size) + 1)
                for i in range(max_num_samples):
                    sample_start = i * config.hop_size
                    sample_end = sample_start + config.segment_length
                    assert sample_end <= len(waveform)

                    # find the lyrics within (start, end)
                    idx_first_word = bisect.bisect_left(start_times, sample_start / config.sr)
                    idx_last_word = bisect.bisect_right(end_times, sample_end / config.sr) - 1

                    if idx_first_word > idx_last_word:  # no words (fully contained) in this sample, skip
                        continue

                    spec = wav2spec(waveform[sample_start:sample_end])     
                    sample = (spec, song['words'][idx_first_word:idx_last_word + 1], song['phowords'][idx_first_word:idx_last_word + 1])
                    samples.append(sample)

            with open(pickle_file, 'wb') as f:
                print(f'Writing {partition} samples')
                pickle.dump(samples, f)

        with open(pickle_file, 'rb') as f:
            print(f'Loading {partition} samples')
            self.samples = pickle.load(f)

    def __getitem__(self, index):
        return self.samples[index]  # (spec, words, phowords)

    def __len__(self):
        return len(self.samples)


class NegativeSampler:
    def __init__(self, dataset):
        # do not store frequencies in a file, they depend on mutable config fields, e.g., use_chars, context

        print('Computing negative sampling probabilities')

        assert config.context <= 1
        self.frequencies = np.zeros((pow(config.vocab_size, 1 + 2 * config.context),), dtype=int)

        for song in tqdm(dataset):

            if config.use_chars:
                tokens = encode_words(song['words'], space_padding=config.context)
            else:
                tokens = encode_phowords(song['phowords'], space_padding=config.context)

            for token in tokens:
                idx = self._token2idx(token)
                self.frequencies[idx] += 1
        

    def sample(self, num_samples, positives, positives_per_scpetrogram):
        # to avoid sampling positives set their frequencies to 0

        negatives = []
        cumsum = np.cumsum([0] + positives_per_scpetrogram)

        for i in range(len(positives_per_scpetrogram)):
            j, k = cumsum[i], cumsum[i + 1]

            # set frequencies of positives to 0
            mutable_frequencies = self.frequencies.copy()
            for l in range(j, k):
                token = positives[l]
                idx = self._token2idx(token)
                mutable_frequencies[idx] = 0
            
            # sample negatives
            prob = mutable_frequencies / np.sum(mutable_frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            negatives += [self._idx2token(idx) for idx in indices]

        return negatives
    
    def fast_sample(self, num_samples, positives, positives_per_scpetrogram):
        # to avoid sampling positives set their frequencies to 0
        
        negatives = []
        cumsum = np.cumsum([0] + positives_per_scpetrogram)

        for i in range(len(positives_per_scpetrogram)):
            j, k = cumsum[i], cumsum[i + 1]

            # set frequencies of positives to 0
            original_idx_freq_pairs = []
            for l in range(j, k):
                token = positives[l]
                idx = self._token2idx(token)
                original_idx_freq_pairs.append((idx, self.frequencies[idx]))
                self.frequencies[idx] = 0

            # sample negatives
            prob = self.frequencies / np.sum(self.frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            negatives += [self._idx2token(idx) for idx in indices]

            # restore original frequencies
            for l in reversed(range(j, k)):  
                # reversed necessary, without, second appearence of same token overwrites true freq with 0 (second appearence gets 0 freq in original_idx_freq_pairs)
                idx, freq = original_idx_freq_pairs[l - j]
                self.frequencies[idx] = freq

        return negatives

    @staticmethod
    def _token2idx(token):
        idx = 0
        for t in token:
            idx *= config.vocab_size
            idx += t
        return idx

    @staticmethod
    def _idx2token(idx):
        token = []
        for _ in range(1 + 2 * config.context):
            token.append(idx % config.vocab_size)
            idx = idx // config.vocab_size
        token = list(reversed(token))
        return token
    


if __name__ == '__main__':
    print('Running data.py')
    
    dali = get_dali()
    print('Size of DALI:', len(dali))
    dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = DaliDataset(dali_train, 'train')
    val_data = DaliDataset(dali_val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))
