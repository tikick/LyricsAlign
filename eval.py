import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

import config
from data import get_dali, get_jamendo, DaliDataset, LyricsDatabase, collate, jamendo_collate
from models import SimilarityModel, contrastive_loss
from utils import set_seed, count_parameters
from decode import align
from time_report import TimeReport


def evaluate(model, device, jamendo):  #, metric='PCO'):
    model.eval()
    PCO_score = 0.
    AAE_score = 0.

    with torch.no_grad():
        for song in tqdm(jamendo):

            config.time_report.start_timer('jamendo_collate')
            spectrogram, positives = jamendo_collate(song)
            torch.cuda.synchronize()
            config.time_report.end_timer('jamendo_collate')

            spectrogram, positives = spectrogram.to(device), positives.to(device)

            config.time_report.start_timer('model')
            S = model(spectrogram, positives)
            torch.cuda.synchronize()
            config.time_report.end_timer('model')
            S = S.cpu()  # detach?
            print(S.shape)

            config.time_report.start_timer('alignment')
            alignment = align(S, song, masked=True)
            torch.cuda.synchronize()
            config.time_report.end_timer('alignment')

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

    config.time_report = TimeReport()

    config.time_report.add_timer('get_jamendo')
    config.time_report.add_timer('eval')
    config.time_report.add_timer('jamendo_collate')
    config.time_report.add_timer('model')
    config.time_report.add_timer('alignment')

    config.time_report.add_timer('DP')
    config.time_report.add_timer('backtracing')
    config.time_report.add_timer('word_alignment')
    config.time_report.add_timer('compute_line_mask')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)

    config.time_report.start_timer('get_jamendo')
    jamendo = get_jamendo()
    torch.cuda.synchronize()
    config.time_report.end_timer('get_jamendo')
    jamendo = jamendo[:1]

    config.time_report.start_timer('eval')
    PCO_score, AAE_score = evaluate(model, device, jamendo)
    torch.cuda.synchronize()
    config.time_report.end_timer('eval')

    config.time_report.report()