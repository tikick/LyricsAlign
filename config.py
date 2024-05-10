import os


## debug
time_report = None


## paths
base_path = '/Users/timonkick/Documents/GitHub/LyricsAlign'  #'/itet-stor/tikick/net_scratch/LyricsAlign'

dali_annotations = os.path.join(base_path, 'DALI_v2.0/annot')
dali_audio = os.path.join(base_path, 'DALI_v2.0/audio')

checkpoint_dir = os.path.join(base_path, 'checkpoints')
pickle_dir = os.path.join(base_path, 'pickles')

jamendo_base = os.path.join(base_path, 'jamendolyrics')
jamendo_metadata = os.path.join(jamendo_base, 'JamendoLyrics.csv')
jamendo_annotations = os.path.join(jamendo_base, 'annotations/words')
jamendo_lyrics = os.path.join(jamendo_base, 'lyrics')
jamendo_audio = os.path.join(jamendo_base, 'mp3')


## model hyperparameters
embedding_dim = 64

##### audio encoder
num_RCBs = 2
channels = 64

##### text encoder
context = 1
use_chars = False  # if false uses phonemes
vocab_size = 28 if use_chars else 40

##### optimizer, data loader and others
num_epochs = 5
lr = 1e-4
batch_size = 64
num_workers = 0
val_size = 0.2  # train validation split


## dataset, samples and spectrograms
num_negative_samples = 1_000
# if you change the following parameters remember to delete the sample files, or the new samples will not be computed
sr = 11025  # waveform sampling rate
segment_length = sr * 5  # length in waveform samples, corresponds to a 5 seconds audio segment
hop_size = segment_length // 2  # in waveform samples
n_fft = 512
fourier_bins = n_fft // 2 + 1