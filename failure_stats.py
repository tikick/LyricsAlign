import os
import torch
import wandb

import config
from data import get_jamendo, get_jamendoshorts, jamendo_collate
from models import SimilarityModel
from utils import fix_seeds
from decode import get_alignment
import example_decoding
import json


def evaluate(model, device, jamendo):
    #model.eval()
    PCO_score_sum = 0.
    AAE_score_sum = 0.
    word_alignments = []

    file_path = os.path.join(config.base_path, "jamendo_alignments.txt")
    with open(file_path, 'r') as f:
        for line in f:
            # Read each line (which contains a JSON-encoded list)
            word_alignment = json.loads(line.strip())  # Convert JSON string to list
            word_alignments.append(word_alignment)

    with torch.no_grad():
        for song, word_alignment in zip(jamendo, word_alignments):
            #if song['id'] != 'Songwriterz_-_Back_In_Time':
            #    continue
            #spectrogram, positives = jamendo_collate(song)
            #spectrogram, positives = spectrogram.to(device), positives.to(device)

            #S = model(spectrogram, positives)
            #S = S.cpu().numpy()

            #_, word_alignment = get_alignment(S, song, time_measure='seconds')
            #word_alignments.append(word_alignment)
            #continue
        
            PCO_score = percentage_of_correct_onsets(song['words'], word_alignment, song['times'])
            AAE_score = average_absolute_error(word_alignment, song['times'])

            print('\n---------------------')
            print(song['id'])
            print(f'PCO: {PCO_score}, AAE: {AAE_score}')
            print('---------------------\n')

            PCO_score_sum += PCO_score
            AAE_score_sum += AAE_score
    
    #file_path = os.path.join(config.base_path, "jamendo_alignments.txt")
    ## Open the file in write mode
    #with open(file_path, 'w') as f:
    #    # Write each list as a JSON string on a new line
    #    for word_alignment in word_alignments:
    #        json.dump(word_alignment, f)  # Write list as JSON string
    #        f.write('\n')  # Add a newline after each list

    return PCO_score_sum / len(jamendo), AAE_score_sum / len(jamendo)


def average_absolute_error(alignment, gt_alignment):
    assert len(alignment) == len(gt_alignment)
    abs_error = 0.
    for time, gt_time in zip(alignment, gt_alignment):
        abs_error += abs(time[0] - gt_time[0])
    return abs_error / len(alignment)


def percentage_of_correct_onsets(words, alignment, gt_alignment, tol=0.3):
    assert len(alignment) == len(gt_alignment) and len(words) == len(alignment)
    correct_onsets = 0
    for idx, (word, time, gt_time) in enumerate(zip(words, alignment, gt_alignment)):
        if abs(time[0] - gt_time[0]) <= tol:
            correct_onsets += 1
        else:
            print(f'word: {word}\ttime: {time}\tgt_time: {gt_time}')

    return correct_onsets / len(alignment)


if __name__ == '__main__':
    print('Running stats.py')

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
           'box_slack': config.box_slack,
           'loss': config.loss,
           'masked': config.masked,
           'use_dali': config.use_dali,
           'val_size': config.val_size}
    
    print(cfg)
    #os.environ["WANDB__SERVICE_WAIT"] = "300"
    #wandb.init(project='New-Align', config=cfg)

    #device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = SimilarityModel().to(device)
    jamendo = get_jamendo()
    #jamendoshorts = get_jamendoshorts()

    #for epoch in range(3, 4):
    #    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '07-05,10:48', str(epoch))))
    #    PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo)

    PCO_jamendo, AAE_jamendo = evaluate(None, None, jamendo)
    print(f'PCO_jamendo: {PCO_jamendo}, AAE_jamendo: {AAE_jamendo}')
