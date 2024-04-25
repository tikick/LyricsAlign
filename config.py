import os


## debug
time_report = None


## paths
base_path = '/itet-stor/tikick/net_scratch/LyricsAlign'
dali_annot_path = os.path.join(base_path, 'DALI_v2.0/annot')
dali_audio_path = os.path.join(base_path, 'DALI_v2.0/audio')
checkpoint_dir = os.path.join(base_path, 'checkpoints')
hdf_dir = os.path.join(base_path, 'hdfs')
pickle_dir = os.path.join(base_path, 'pickles')


## model hyperparameters
embedding_dim = 64

##### audio encoder
num_RCBs = 5
channels = 64

##### text encoder
context = 1
use_chars = False  # if false uses phonemes
vocab_size = 28 if use_chars else 40

##### optimizer, data loader and others
num_epochs = 3
lr = 1e-4
batch_size = 32
num_workers = 1
val_size = 0.2  # train validation split


## dataset, samples and spectrograms
# if you change these parameters remember to delete the sample files, or the new samples will not be computed
sr = 11025  # waveform sampling rate
sample_length = 11025 * 5  # length in waveform samples
hop_size = sample_length // 2  # in waveform samples
num_negative_samples = 1_000
n_fft = 512
fourier_bins = n_fft // 2 + 1