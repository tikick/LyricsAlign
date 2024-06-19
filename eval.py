import os
import torch
import wandb

import config
from data import get_jamendo, get_jamendoshorts, jamendo_collate
from models import SimilarityModel
from utils import fix_seeds
from decode import align
import example_decoding


def evaluate(model, device, jamendo, log, epoch):
    model.eval()
    PCO_score = 0.
    AAE_score = 0.

    with torch.no_grad():
        for song in jamendo:
            #print(song['id'])
            #print(song['words'])
            #print(song['times'])
            spectrogram, positives = jamendo_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu().numpy()
            #print('S.shape:', S.shape)

            alignment = example_decoding.align(S, song) #align(S, song, level='word', log=log, epoch=epoch)
            PCO_score += percentage_of_correct_onsets(alignment, song['times'])
            AAE_score += average_absolute_error(alignment, song['times'])
        
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

    fix_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)
    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()

    for epoch in range(2):
        #epoch = 11
        model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '06-12,18:50', str(epoch))))

        #evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)

        print('Epoch', epoch)
        print('metric/PCO_jamendo', PCO_jamendo)
        print('metric/AAE_jamendo', AAE_jamendo)
        print('\n---------------------\n')