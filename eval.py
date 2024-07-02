import os
import torch
import wandb

import config
from data import get_jamendo, get_jamendoshorts, jamendo_collate
from models import SimilarityModel
from utils import fix_seeds
from decode import get_alignment
import example_decoding
from media import log_plots


def evaluate(model, device, jamendo, log=False, epoch=-1):
    model.eval()
    PCO_score = 0.
    AAE_score = 0.

    with torch.no_grad():
        for song in jamendo:
            #print(song['id'])
            #print(song['words'])
            #print(song['times'])
            waveform, positives = jamendo_collate(song)
            waveform, positives = waveform.to(device), positives.to(device)

            S = model(waveform, positives)
            S = S.cpu().numpy()
            #print('S.shape:', S.shape)

            _, word_alignment = get_alignment(S, song, time_measure='seconds')
            PCO_score += percentage_of_correct_onsets(word_alignment, song['times'])
            AAE_score += average_absolute_error(word_alignment, song['times'])

            # log plots for inspection
            if log:
                token_alignment, word_alignment = get_alignment(S, song, time_measure='frames')
                log_plots(S, song, token_alignment, word_alignment, epoch)

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

    cfg = {'num_RCBs': config.num_RCBs,
           'channels': config.channels,
           'context': config.context,
           'use_chars': config.use_chars,
           'embedding_dim': config.embedding_dim,
           'num_epochs': config.num_epochs,
           'lr': config.lr,
           'batch_size': config.batch_size,
           'num_negative_samples': config.num_negative_samples,
           'alpha': config.alpha,
           'masked': config.masked,
           'use_dali': config.use_dali,
           'words_slack': config.words_slack,
           'val_size': config.val_size}
    
    print(cfg)
    wandb.init(project='New-Align', config=cfg)

    device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimilarityModel().to(device)
    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()

    for epoch in range(4, 6):
        model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '06-19,10:29', str(epoch))))

        evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False)
        wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
