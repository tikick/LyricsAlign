import os


## general
sr = 11025  # sampling rate
load_model = None  # path to model to load
val_size = 0.2  # train validation split


## paths
base_path = '/Users/timonkick/Documents/ETH/master/Sem3/LyricsAlignment/code'
dali_annot_path = os.path.join(base_path, 'DALI_v2.0/annot')
dali_audio_path = os.path.join(base_path, 'DALI_v2.0/audio')
checkpoint_dir = os.path.join(base_path, 'checkpoints')
log_dir = os.path.join(base_path, 'logs')
pickle_dir = os.path.join(base_path, 'pickles')


## model hyperparameters
embedding_dim = 64

##### audio encoder
channels = 64
input_length = 11025 * 5  # length in samples

##### text encoder
context = 1
use_chars = False
vocab_size = 28 if use_chars else 40

##### optimizer and data loaders
lr = 1e-3
batch_size = 32
num_workers = 8


## dataset and samples
hop_size = input_length // 2  # in samples
num_negative_samples = 1_000
