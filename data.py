# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import DALI as dali_code
import bisect
import os
from tqdm import tqdm
import pickle
import csv
import pandas as pd
from math import floor
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config
from utils import encode_words, encode_phowords, words2phowords, lines2pholines, \
    load, wav2spec, read_jamendo_times, normalize_dali, normalize_georg, normalize_jamendo, \
        monotonically_increasing_starts, monotonically_increasing_ends, old_monotonically_increasing_times, get_dali_remarks


def _get_dali(keep, lang='english'):
    # 96569 of 5069058 chars in DALI are not in utils.char_dict and thus removed in normalize_dali (2% noise)
    # above stat did not consider words completely removed, not up to date

    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[],
        keep=keep)
        #keep=['0a3cd469757e470389178d44808273ab', '0a81772ae3a7404f9ef09ecd1f94db07', '0dea06fa7ca04eb88b17e8d83993adc3', '1ae34dc139ea43669501fb9cef85cbd0', '1afbb77f88dc44e9bedc07b54341be9c', '1b9c139f491c41f5b0776eefd21c122d'])

    songs = []

    for id in keep:
        annot = dali_data[id].annotations['annot']
        metadata = dali_data[id].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue
        
        words = [d['text'] for d in annot['words']]
        times = [d['time'] for d in annot['words']]
        words, times = normalize_dali(words, times)
        phowords = words2phowords(words)  #[d['text'] for d in annot['phonemes']]

        song = {'id': id,
                'audio_path': os.path.join(config.dali_audio, id + '.wav'),
                'words': words,
                'phowords': phowords,
                'times': times}
        
        break
        if dali_song_is_corrupt(song):
            continue

        songs.append(song)

    return songs

def get_non_monotonic_dali(lang='english'):
    # 96569 of 5069058 chars in DALI are not in utils.char_dict and thus removed in normalize_dali (2% noise)
    # above stat did not consider words completely removed, not up to date

    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[], keep=[])

    non_monotonic_dali_songs = []
    old_non_monotonic_dali_songs = []

    def unique(l):
        last = object()
        for item in l:
            if item == last:
                continue
            yield item
            last = item

    num_non_monotonic_ends = 0

    audio_files = os.listdir(config.dali_audio)  # only get songs for which we have audio files
    for file in tqdm(audio_files):
        id = file[:-4]
        annot = dali_data[id].annotations['annot']
        metadata = dali_data[id].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue

        times = [d['time'] for d in annot['words']]
        words = [d['text'] for d in annot['words']]
        words, times = normalize_dali(words, times, cutoff=1e10)
        phowords = words2phowords(words)  #[d['text'] for d in annot['phonemes']]

        song = {'id': id,
                'audio_path': os.path.join(config.dali_audio, file),
                'words': words,
                'phowords': phowords,
                'times': times,
                'url': dali_data[id].info['audio']['url']}

        if not monotonically_increasing_starts(times):
            ### sort and remove duplicates
            times, words, phowords = (list(t) for t in zip(*unique(sorted((zip(times, words, phowords))))))
            song['times'] = times
            song['words'] = words
            song['phowords'] = phowords
            ###
            non_monotonic_dali_songs.append(song)
        elif not old_monotonically_increasing_times(times):
            old_non_monotonic_dali_songs.append(song)

        if not monotonically_increasing_ends(times):
            num_non_monotonic_ends += 1

    print(f'num_non_monotonic_ends = {num_non_monotonic_ends}')

    return non_monotonic_dali_songs, old_non_monotonic_dali_songs

def get_dali(lang='english'):
    # 96569 of 5069058 chars in DALI are not in utils.char_dict and thus removed in normalize_dali (2% noise)
    # above stat did not consider words completely removed, not up to date

    dali_data = dali_code.get_the_DALI_dataset(config.dali_annotations, skip=[], keep=[])
    remarks = get_dali_remarks()

    songs = []
    corrupt = 0
    
    def unique(l):
        last = object()
        for item in l:
            if item == last:
                continue
            yield item
            last = item

    corrupt = 0
    audio_files = os.listdir(config.dali_audio)  # only get songs for which we have audio files
    for file in tqdm(audio_files):
        id = file[:-4]
        annot = dali_data[id].annotations['annot']
        metadata = dali_data[id].info['metadata']

        if lang is not None and metadata['language'] != lang:
            continue
        
        offset = 0
        cutoff = 1e10
        if id in remarks:
            if remarks[id]['corrupt from'] == 0 or remarks[id]['noisy'] or remarks[id]['offset'].__contains__(',') or remarks[id]['non-english']:
                corrupt += 1
                continue
            offset = 0 if remarks[id]['offset'] in '+-' else float(remarks[id]['offset'])
            cutoff = remarks[id]['corrupt from']

        times = [d['time'] for d in annot['words']]
        words = [d['text'] for d in annot['words']]
        words, times = normalize_dali(words, times, cutoff, offset)
        phowords = words2phowords(words)  #[d['text'] for d in annot['phonemes']]

        if not monotonically_increasing_starts(times):
            # sort and remove duplicates
            times, words, phowords = (list(t) for t in zip(*unique(sorted((zip(times, words, phowords))))))

        song = {'id': id,
                'audio_path': os.path.join(config.dali_audio, file),
                'words': words,
                'phowords': phowords,
                'times': times,
                'url': dali_data[id].info['audio']['url']}

        songs.append(song)

    print(f'corrupt = {corrupt}')
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


def get_georg():
    # num_unk_chars = 39330, num_total_chars = 19920586, alignment_nones = 755
    songs = []
    
    for i in range(20):  # for folders from 0 to 19
        parq_file = os.path.join(config.georg_annotations, str(i), 'alignment.parq')

        df = pd.read_parquet(parq_file, engine='pyarrow')
        for _, row in df.iterrows():

            if row['alignment'] is None:
                continue

            audio_path = os.path.join(config.georg_audio, row['ytid'] + '.mp3')
            if not os.path.exists(audio_path):
                continue

            token_starts = row['alignment']['starts']
            token_ends = row['alignment']['ends']
            tokens_per_word = list(row['alignment']['tokens_per_word'])
            token_offsets = np.cumsum([0] + tokens_per_word)
            assert token_offsets[-1] == len(token_starts)

            word_starts = []
            word_ends = []
            for token_offset in token_offsets[:-1]:
                word_starts.append(token_starts[token_offset])
            for token_offset in token_offsets[1:]:
                word_ends.append(token_ends[token_offset - 1])

            times = list(zip(word_starts, word_ends))
            words = row['alignment']['words']
            words, times = normalize_georg(words, times)
            phowords = words2phowords(words)
            
            song = {'id': row['ytid'],
                    'audio_path': audio_path,
                    'words': words,
                    'phowords': phowords,
                    'times': times}
            
            #if georg_song_is_corrupt(song):
            #    continue
            
            songs.append(song)
    
    return songs


def jamendo_collate(song):
    waveform = load(song['audio_path'], sr=config.sr)
    song['duration'] = len(waveform) / config.sr
    spec = wav2spec(waveform)
    spectrogram, all_tokens, _, _ = collate(data=[(spec, song['words'], song['phowords'], song['times'])])
    return spectrogram, all_tokens


def collate(data):
    spectrograms = []
    all_tokens = []
    all_times = []
    tokens_per_spectrogram = []

    for spec, words, phowords, times in data:
        spectrograms.append(spec)

        if config.use_chars:
            tokens, token_times = encode_words(words, times)
        else:
            tokens, token_times = encode_phowords(phowords, times)

        tokens_per_spectrogram.append(len(tokens))
        all_tokens += tokens
        all_times += token_times

    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Convert the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    spectrograms = torch.Tensor(np.array(spectrograms))
    all_tokens = torch.IntTensor(all_tokens)

    return spectrograms, all_tokens, all_times, tokens_per_spectrogram


class LA_Dataset(Dataset):
    def __init__(self, dataset, partition):
        super(LA_Dataset, self).__init__()
        dataset_name = 'clean_monotonic_dali' if config.use_dali else 'georg'
        file_name = f'{dataset_name}_{partition}_with_time'
        #file_name = f'{dataset_name}_{partition}_100'
        pickle_file = os.path.join(config.pickle_dir, file_name + '.pkl')

        if not os.path.exists(pickle_file):
            if not os.path.exists(config.pickle_dir):
                os.makedirs(config.pickle_dir)

            print(f'Creating {file_name} samples')
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
                    idx_past_last_word = idx_first_word  #bisect.bisect_left(end_times, sample_end / config.sr)
                    while idx_past_last_word < len(end_times) and end_times[idx_past_last_word] < sample_end / config.sr:
                        idx_past_last_word += 1

                    if idx_first_word >= idx_past_last_word:  # no words (fully contained) in this sample, skip
                        continue
                    
                    # sample spectrogram, words, phowords and relative times withing sample
                    spec = wav2spec(waveform[sample_start:sample_end])
                    words = song['words'][idx_first_word:idx_past_last_word]
                    phowords = song['phowords'][idx_first_word:idx_past_last_word]
                    times = song['times'][idx_first_word:idx_past_last_word]
                    offset = sample_start / config.sr
                    times = [(start - offset, end - offset) for (start, end) in times]
                    for (start, end) in times:
                        # +0.03 due to georg's failed monotonicity
                        #assert 0 <= start < 5.03 and 0 <= end < 5.03, f'id={song["id"]}, i={i}, sample_start={sample_start}, offset={offset} start={start}, end={end}'
                        assert 0 <= start < end < 5, f'id={song["id"]}, i={i}, sample_start={sample_start}, offset={offset} start={start}, end={end}'
                    sample = (spec, words, phowords, times)
                    samples.append(sample)

            with open(pickle_file, 'wb') as f:
                print(f'Writing {file_name} samples')
                pickle.dump(samples, f)

        with open(pickle_file, 'rb') as f:
            print(f'Loading {file_name} samples')
            self.samples = pickle.load(f)

    def __getitem__(self, index):
        return self.samples[index]  # (spec, words, phowords, times)

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
                tokens, _ = encode_words(song['words'], song['times'])
            else:
                tokens, _ = encode_phowords(song['phowords'], song['times'])

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
    
    #georg = get_georg()
    #print('Size of Georg:', len(georg))
    dali = get_dali()
    print('Size of DALI:', len(dali))
    train, val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = LA_Dataset(train, 'train')
    val_data = LA_Dataset(val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))
