import os


## debug
train = False
time_report = None



## paths
base_path = '/itet-stor/tikick/net_scratch/LyricsAlign'#'/Users/timonkick/Documents/GitHub/LyricsAlign'

dali_annotations = os.path.join(base_path, 'DALI_v2.0/annot')
dali_audio = os.path.join(base_path, 'DALI_v2.0/wav_audio')

checkpoint_dir = os.path.join(base_path, 'checkpoints')
pickle_dir = os.path.join(base_path, 'pickles')

jamendo_base = os.path.join(base_path, 'jamendolyrics')
jamendo_metadata = os.path.join(jamendo_base, 'JamendoLyrics.csv')
jamendo_annotations = os.path.join(jamendo_base, 'annotations/words')
jamendo_lyrics = os.path.join(jamendo_base, 'lyrics')
jamendo_audio = os.path.join(jamendo_base, 'mp3')

jamendo_segments_base = os.path.join(base_path, 'jamendo_segments')
jamendo_segments_metadata = os.path.join(jamendo_segments_base, 'JamendoSegments.csv')
jamendo_segments_annotations = os.path.join(jamendo_segments_base, 'annotations/words')
jamendo_segments_lyrics = os.path.join(jamendo_segments_base, 'lyrics')
jamendo_segments_audio = os.path.join(jamendo_segments_base, 'mp3')



## model hyperparameters
embedding_dim = 64

##### audio encoder
num_RCBs = 5
channels = 64

##### text encoder
context = 1
use_chars = True  # if false uses phonemes
vocab_size = 28 if use_chars else 40

##### optimizer, data loader and others
num_epochs = 10
lr = 1e-4
batch_size = 64


## dataset, samples and spectrograms
num_negative_samples = 1_000
# if you change the following parameters remember to delete the sample files, or the new samples will not be computed
val_size = 0.1  # train validation split
sr = 11025  # waveform sampling rate
segment_length = sr * 5  # length in waveform samples, corresponds to a 5 seconds audio segment
hop_size = segment_length // 2  # in waveform samples
n_fft = 512
fourier_bins = n_fft // 2 + 1


## alignment
masked = False
