# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime

import config
from data import get_dali, get_jamendo, DaliDataset, LyricsDatabase, collate, get_jamendo_segments
from models import SimilarityModel, contrastive_loss
from utils import fix_seed, count_parameters, int2char, int2phoneme
from eval import evaluate
from decode import show


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.
    batch_loss = 0.

    for idx, batch in enumerate(tqdm(train_loader)):
        spectrograms, positives, len_positives = batch
        negatives = lyrics_database.sample(config.num_negative_samples, positives, len_positives)
        negatives = torch.IntTensor(negatives)
        spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

        optimizer.zero_grad()

        PA, NA = model(spectrograms, positives, len_positives, negatives)

        loss = criterion(PA, NA)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        batch_loss += loss.item()

        log_checkpoint = 100  # log every 100 batches
        if (idx + 1) % log_checkpoint == 0:
            wandb.log({'train/batch_loss': batch_loss / log_checkpoint, 'train/batch_idx': idx + 1})
            batch_loss = 0.

    return train_loss / num_batches


def validate(model, device, val_loader, lyrics_database, criterion, epoch):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            spectrograms, positives, len_positives = batch
            negatives = lyrics_database.sample(config.num_negative_samples, positives, len_positives)
            negatives = torch.IntTensor(negatives)
            spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

            PA, NA = model(spectrograms, positives, len_positives, negatives)

            loss = criterion(PA, NA)
            val_loss += loss.item()

            # log first batch
            if idx == 0:
                PA = PA.cpu().numpy()
                NA = NA.cpu().numpy()
                #PA = 0.5 * (PA + 1)
                #NA = 0.5 * (NA + 1)
                positive_tokens = positives.cpu().tolist()
                negative_tokens = negatives.cpu().tolist()

                #m = len(positive_tokens[0]) // 2
                f = int2char if config.use_chars else int2phoneme
                positive_tokens = [[f[pos_token[i]] for i in range(len(pos_token))] for pos_token in positive_tokens]
                negative_tokens = [[f[neg_token[i]] for i in range(len(neg_token))] for neg_token in negative_tokens]

                cumsum = np.cumsum([0] + len_positives)
                for i in range(8):
                    heights = [len_positives[i], 50]
                    fig, axs = plt.subplots(2, 1, height_ratios=heights, 
                                            figsize=(15, min((sum(heights) + 20 * len(heights)) // 12, 100)))

                    j, k = cumsum[i], cumsum[i + 1]
                    show(PA[j:k], axs[0], 'positive scores', positive_tokens[j:k])  # PA[i]
                    j = i * config.num_negative_samples
                    k = j + 50  # don't show all 1000 negative tokens
                    show(NA[j:k], axs[1], 'negative scores', negative_tokens[j:k])  # NA[i]

                    fig.tight_layout()

                    wandb.log({'media/DALI_sample_' + str(i): plt, 'media/epoch': epoch})
                    plt.close()

    return val_loss / num_batches


def main():
    fix_seed()

    cfg = {'embedding_dim': config.embedding_dim,
           'num_RCBs': config.num_RCBs,
           'channels': config.channels,
           'context': config.context,
           'use_chars': config.use_chars,
           'num_epochs': config.num_epochs,
           'lr': config.lr,
           'batch_size': config.batch_size,
           'num_negative_samples': config.num_negative_samples,
           'val_size': config.val_size,
           'masked': config.masked}
           #'dali_size': len(dali)}
    
    print(cfg)
    wandb.init(project='Train-Decode', config=cfg)

    start_time_run = datetime.now().strftime('%m-%d,%H:%M')
    run_checkpoint_dir = os.path.join(config.checkpoint_dir, start_time_run)
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.isdir(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device:', device)

    model = SimilarityModel().to(device)
    # count_parameters(model)

    # if train and val files already exist:
        # dali = dali_train = dali_val = []  # no need to load dali, files already exist
    # else:
    dali = get_dali()
    print('Size of DALI:', len(dali))
    dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = DaliDataset(dali_train, 'train')
    val_data = DaliDataset(dali_val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    lyrics_database = LyricsDatabase(dali)

    jamendo = get_jamendo()
    jamendo_segments = get_jamendo_segments()
    dali_val_20 = dali_val[:20]
    dali_train_20 = dali_train[:20]

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    # log before training
    epoch = -1

    config.train = False
    val_loss = validate(model, device, val_loader, lyrics_database, criterion, epoch)
    wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

    PCO_jamendo_segments, AAE_jamendo_segments = evaluate(model, device, jamendo_segments, log=True, epoch=epoch)
    PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)
    PCO_dali_val_20, AAE_dali_val_20 = evaluate(model, device, dali_val_20, log=False, epoch=-1)
    PCO_dali_train_20, AAE_dali_train_20 = evaluate(model, device, dali_train_20, log=False, epoch=-1)
    wandb.log({'metric/PCO_jamendo_segments': PCO_jamendo_segments, 'metric/epoch': epoch})
    wandb.log({'metric/AAE_jamendo_segments': AAE_jamendo_segments, 'metric/epoch': epoch})
    wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
    wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
    wandb.log({'metric/PCO_dali_val_20': PCO_dali_val_20, 'metric/epoch': epoch})
    wandb.log({'metric/AAE_dali_val_20': AAE_dali_val_20, 'metric/epoch': epoch})
    wandb.log({'metric/PCO_dali_train_20': PCO_dali_train_20, 'metric/epoch': epoch})
    wandb.log({'metric/AAE_dali_train_20': AAE_dali_train_20, 'metric/epoch': epoch})

    #model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '05-31,19:48', '6')))
    epoch = 0#7
    while epoch < config.num_epochs:
        print('Epoch:', epoch)

        config.train = True
        train_loss = train(model, device, train_loader, lyrics_database, criterion, optimizer)
        wandb.log({'train/train_loss': train_loss, 'train/epoch': epoch})

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, str(epoch)))

        config.train = False
        val_loss = validate(model, device, val_loader, lyrics_database, criterion, epoch)
        wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

        PCO_jamendo_segments, AAE_jamendo_segments = evaluate(model, device, jamendo_segments, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)
        PCO_dali_val_20, AAE_dali_val_20 = evaluate(model, device, dali_val_20, log=False, epoch=-1)
        PCO_dali_train_20, AAE_dali_train_20 = evaluate(model, device, dali_train_20, log=False, epoch=-1)
        wandb.log({'metric/PCO_jamendo_segments': PCO_jamendo_segments, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo_segments': AAE_jamendo_segments, 'metric/epoch': epoch})
        wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/PCO_dali_val_20': PCO_dali_val_20, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_dali_val_20': AAE_dali_val_20, 'metric/epoch': epoch})
        wandb.log({'metric/PCO_dali_train_20': PCO_dali_train_20, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_dali_train_20': AAE_dali_train_20, 'metric/epoch': epoch})

        print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:3f}, PCO: {PCO_jamendo}, AAE: {AAE_jamendo}')

        epoch += 1

    wandb.finish()


if __name__ == '__main__':
    print('Running train.py')
    main()
