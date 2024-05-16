import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

import config
from data import get_dali, get_jamendo, DaliDataset, LyricsDatabase, collate, jamendo_collate, get_jamendo_segments
from models import SimilarityModel, contrastive_loss
from utils import fix_seed, count_parameters
from decode import align, _align


def evaluate(model, device, jamendo):  #, metric='PCO'):
    model.eval()
    PCO_score = 0.
    AAE_score = 0.

    with torch.no_grad():
        for song in tqdm(jamendo):
            spectrogram, positives = jamendo_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu().numpy()
            #print('S.shape:', S.shape)

            alignment = _align(S, song)  # align
            PCO_score += percentage_of_correct_onsets(alignment, song['gt_alignment'])
            AAE_score += average_absolute_error(alignment, song['gt_alignment'])
        
    return PCO_score / len(jamendo), AAE_score / len(jamendo)


def average_absolute_error(alignment, gt_alignment):
    assert len(alignment) == len(gt_alignment)
    abs_error = 0.
    for time, gt_time in zip(alignment, gt_alignment):
        abs_error += abs(time[0] - gt_time[0])
    return abs_error / len(alignment)


def percentage_of_correct_onsets(alignment, gt_alignment, tol=0.3):
    assert len(alignment) == len(gt_alignment)
    correct_onsets = 0
    for time, gt_time in zip(alignment, gt_alignment):
        if abs(time[0] - gt_time[0]) <= tol:
            correct_onsets += 1
    return correct_onsets / len(alignment)


def segment():
    sr = 44100
    audio_file = 'Tom_Orlando_-_The_One__feat._Tina_G_.mp3'

    start = 18
    end = 23

    audio = load(os.path.join(config.jamendo_audio, audio_file), sr=sr)
    audio_segment = audio[start * sr:end * sr]
    #save_path = os.path.join(config.jamendo_segments_audio, str(start) + '-' + str(end) + '_' + audio_file)
    #sf.write(save_path, segment, sr)

    with open(os.path.join(config.jamendo_lyrics, audio_file[:-4] + '.txt'), 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if len(l) > 0]  # remove empty lines between paragraphs
    words = ' '.join(lines).split()
    times = read_gt_alignment(os.path.join(config.jamendo_annotations, audio_file[:-4] + '.csv'))
    segment_words = []
    segment_times = []
    for word, time in zip(words, times):
        if time[0] >= start and time[1] <= end:
            segment_words.appen(word)
            segment_times.append((time[0] - start, time[1] - start))

    return audio_segment, segment_words, segment_times

if __name__ == '__main__':
    print('Running eval.py')

    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'checkpoint_9')))
    #jamendo = get_jamendo()
    #jamendo = jamendo[:1]
    jamendo = get_jamendo_segments()

    wandb.init(project='Decode')
    PCO_score, AAE_score = evaluate(model, device, jamendo)
    wandb.finish()