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
from utils import fix_seed, count_parameters, load, read_gt_alignment
from decode import align, _align


def evaluate(model, device, jamendo):  #, metric='PCO'):
    model.eval()
    PCO_score = 0.
    AAE_score = 0.

    with torch.no_grad():
        for song in jamendo:
            print(f'song:\n{song['id']}\n{song['words']}\n{song['gt_alignment']}')
            spectrogram, positives = jamendo_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu().numpy()
            print('S.shape:', S.shape)

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


if __name__ == '__main__':
    print('Running eval.py')

    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'checkpoint_9')))
    jamendo = get_jamendo_segments()#get_jamendo()
    jamendo = jamendo[:1]

    wandb.init(project='Decode')
    PCO_score, AAE_score = evaluate(model, device, jamendo)
    wandb.finish()