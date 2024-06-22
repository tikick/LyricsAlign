import os


# paths
base_path = '/itet-stor/tikick/net_scratch/LyricsAlign' #'/Users/timonkick/Documents/GitHub/LyricsAlign'

checkpoint_dir = os.path.join(base_path, 'checkpoints')
pickle_dir = os.path.join(base_path, 'pickles')

dali_base = os.path.join(base_path, 'DALI_v2.0')
dali_annotations = os.path.join(dali_base, 'annot')
dali_audio = os.path.join(dali_base, 'wav')

jamendo_base = os.path.join(base_path, 'jamendolyrics')
jamendo_metadata = os.path.join(jamendo_base, 'JamendoLyrics.csv')
jamendo_annotations = os.path.join(jamendo_base, 'annotations/words')
jamendo_lyrics = os.path.join(jamendo_base, 'lyrics')
jamendo_audio = os.path.join(jamendo_base, 'mp3')

jamendoshorts_base = os.path.join(base_path, 'jamendoshorts')
jamendoshorts_metadata = os.path.join(jamendoshorts_base, 'JamendoShorts.csv')
jamendoshorts_annotations = os.path.join(jamendoshorts_base, 'annotations/words')
jamendoshorts_lyrics = os.path.join(jamendoshorts_base, 'lyrics')
jamendoshorts_audio = os.path.join(jamendoshorts_base, 'mp3')

georg_base = os.path.join(base_path, 'Georg')
georg_annotations = os.path.join(georg_base, 'ttv')
georg_audio = os.path.join(georg_base, 'data/audio')


# hyperparameters

### audio encoder
num_RCBs = 10
channels = 64

### text encoder
context = 1  # when context = 0, for some audio segments the set of negatives might be empty -> NegativeSampler.sample() fails
use_chars = False  # if false uses phonemes
vocab_size = 28 if use_chars else 40  # len(char_dict) if use_chars else len(phoneme_dict)

### audio and text encoder
embedding_dim = 64

### others:
num_epochs = 4
lr = 0.0001
batch_size = 8
num_negative_samples = 1_000


# loss
alpha = 0.5


# alignment
masked = False


# dataset
use_dali = False  # if false uses georg


# WARNING: if you change the following parameters remember to delete the sample files or the new samples will not be computed

# dataset
words_slack = 0  # words identified within audio segment have a words_slack slack (in seconds) to the true audio segment boundary.
val_size = 0.1

# waveform
sr = 11025  # waveform sampling rate
segment_length = sr * 5  # length in waveform samples, corresponds to a 5 seconds audio segment
hop_size = segment_length // 2  # in waveform samples

# spectrogram
n_fft = 512
freq_bins = n_fft // 2 + 1
