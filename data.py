# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import DALI as dali_code
import bisect
import numpy as np
import os
from tqdm import tqdm
import pickle
from math import floor
import torch
from torch.utils.data import Dataset
import csv
from sklearn.model_selection import train_test_split

import config
from utils import encode_words, encode_phowords, load, wav2spec, words2phowords, lines2pholines, read_gt_alignment, normalize_dali_annot


def get_dali(lang='english'):
    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[],
        keep=[])
        #keep=['0a3cd469757e470389178d44808273ab', '0a81772ae3a7404f9ef09ecd1f94db07', '0dea06fa7ca04eb88b17e8d83993adc3', '1ae34dc139ea43669501fb9cef85cbd0', '1afbb77f88dc44e9bedc07b54341be9c', '1b9c139f491c41f5b0776eefd21c122d'])

    songs = []

    audio_files = os.listdir(config.dali_audio)  # only get songs for which we have audio files
    unk_chars = 0
    total_chars = 0  # measure DALI noise
    for file in audio_files:
        annot = dali_data[file[:-4]].annotations['annot']
        metadata = dali_data[file[:-4]].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue
        
        words = [d['text'] for d in annot['words']]
        times = [d['time'] for d in annot['words']]
        words, times, _unk_chars, _total_chars = normalize_dali_annot(words, times)#, cut=config.dali_cut_words_with_unknown_chars)
        phowords = words2phowords(words)  #[d['text'] for d in annot['phonemes']]
        unk_chars += _unk_chars
        total_chars += _total_chars

        song = {'id': file[:-4],
                'audio_path': os.path.join(config.dali_audio, file),
                'words': words,
                'phowords': phowords,
                'times': times
                }

        songs.append(song)

    print(f'DALI num unknown chars = {unk_chars}, DALI total chars = {total_chars}')

    return songs


def get_jamendo(lang='English'):  # jamendo is already normalized
    songs = []

    with open(config.jamendo_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendo_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = [l for l in lines if len(l) > 0]  # remove empty lines between paragraphs
            words = ' '.join(lines).split()
            phowords = words2phowords(words)
            pholines = lines2pholines(lines)
            gt_alignment = read_gt_alignment(os.path.join(config.jamendo_annotations, audio_file[:-4] + '.csv'))
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendo_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': gt_alignment
                    }
            
            songs.append(song)
    
    return songs


def get_jamendo_segments(lang='English'):  # jamendo is already normalized
    songs = []

    with open(config.jamendo_segments_metadata, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Language'] != lang:
                continue

            audio_file = row['Filepath']
            with open(os.path.join(config.jamendo_segments_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
                lines = f.read().splitlines()
            lines = [l for l in lines if len(l) > 0]  # remove empty lines between paragraphs
            words = ' '.join(lines).split()
            phowords = words2phowords(words)
            pholines = lines2pholines(lines)
            gt_alignment = read_gt_alignment(os.path.join(config.jamendo_segments_annotations, audio_file[:-4] + '.csv'))
            
            song = {'id': audio_file[:-4],
                    'audio_path': os.path.join(config.jamendo_segments_audio, audio_file),
                    'words': words,
                    'phowords': phowords,
                    'lines': lines,
                    'pholines': pholines,
                    'times': gt_alignment
                    }
            
            songs.append(song)
    
    return songs


def jamendo_collate(song):
    waveform = load(song['audio_path'], sr=config.sr)
    spec = wav2spec(waveform)
    spectrogram, contextual_tokens, _ = collate(data=[(spec, song['words'], song['phowords'])], eval=True)
    return spectrogram, contextual_tokens


def collate(data, eval=False):
    spectrograms = []
    contextual_tokens = []
    len_tokens = []

    for spec, words, phowords in data:
        spectrograms.append(spec)

        padding = config.context
        if eval:
            padding += 1  # +1 silence padding for alignment
        if config.use_chars:
            tokens = encode_words(words, space_padding=padding)
        else:
            tokens = encode_phowords(phowords, space_padding=padding)

        # extract context for each token
        token_with_context = [tokens[i:i + 2 * config.context + 1] for i in range(len(tokens) - 2 * config.context)]
        contextual_tokens += token_with_context
        assert len(token_with_context) > 0
        len_tokens.append(len(token_with_context))

    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    spectrograms = torch.Tensor(np.array(spectrograms))
    contextual_tokens = torch.IntTensor(contextual_tokens)

    return spectrograms, contextual_tokens, len_tokens


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


class LyricsDatabase:
    def __init__(self, dataset):

        pickle_file = os.path.join(config.pickle_dir,
                                   'dali_neg_probs-' + 'char' if config.use_chars else 'phoneme' + '.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print('Computing negative sampling probabilities')

            assert config.context <= 1
            frequencies = np.zeros((pow(config.vocab_size, 2 * config.context + 1),), dtype=int)

            for song in tqdm(dataset):

                if config.use_chars:
                    tokens = encode_words(song['words'], space_padding=config.context)
                else:
                    tokens = encode_phowords(song['phowords'], space_padding=config.context)

                # extract context for each token
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
            original_freq = self.frequencies.copy() #[]
            for l in range(j, k):
                contextual_token = pos[l]
                idx = self._contextual_token2idx(contextual_token)
                #original_freq.append(self.frequencies[idx])
                assert self.frequencies[idx] > 0
                self.frequencies[idx] = 0
            
            # sample negatives
            prob = self.frequencies / np.sum(self.frequencies)
            indices = np.random.choice(len(prob), size=num_samples, p=prob)
            contextual_tokens += [self._idx2contextual_token(idx) for idx in indices]

            # restore original frequencies
            self.frequencies = original_freq.copy()
            #for l in range(j, k):
            #    contextual_token = pos[l]
            #    idx = self._contextual_token2idx(contextual_token)
            #    self.frequencies[idx] = original_freq[l - j]

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
    


if __name__ == '__main__':
    print('Running data.py')
    
    dali = get_dali()
    print('Size of DALI:', len(dali))
    dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = DaliDataset(dali_train, 'train')
    val_data = DaliDataset(dali_val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))
    
    lyrics_database = LyricsDatabase(dali)