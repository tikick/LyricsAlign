# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

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
from eval import evaluate
from time_report import TimeReport


def contrastive_loss(PA, NA):
    return torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
           torch.mean(torch.pow(torch.max(NA, dim=1).values, 2))  # max along time dimension


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    #num_batches = 1

    train_loader = iter(train_loader)

    config.time_report.start_timer('epoch')

    for _ in tqdm(range(num_batches)):

        config.time_report.start_timer('train_loader')
        batch = next(train_loader)
        torch.cuda.synchronize()  # before end_timer
        config.time_report.end_timer('train_loader')

        spectrograms, positives, len_positives = batch

        config.time_report.start_timer('sampling negatives')
        negatives = lyrics_database.sample(config.num_negative_samples, positives, len_positives)
        torch.cuda.synchronize()
        config.time_report.end_timer('sampling negatives')
        
        negatives = torch.IntTensor(negatives)

        #print('data.shape:', spectrograms.shape, positives.shape, negatives.shape)

        config.time_report.start_timer('batch.to(device)')
        spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)
        torch.cuda.synchronize()
        config.time_report.end_timer('batch.to(device)')

        optimizer.zero_grad()

        config.time_report.start_timer('model')
        PA, NA = model(spectrograms, positives, len_positives, negatives)
        torch.cuda.synchronize()
        config.time_report.end_timer('model')

        config.time_report.start_timer('loss')
        loss = criterion(PA, NA)
        torch.cuda.synchronize()
        config.time_report.end_timer('loss')

        config.time_report.start_timer('backward')
        loss.backward()
        torch.cuda.synchronize()
        config.time_report.end_timer('backward')

        optimizer.step()

    torch.cuda.synchronize()
    config.time_report.end_timer('epoch')


def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device:', device)

    model = SimilarityModel().to(device)
    # count_parameters(model)

    # if train and val files already exist:
        # dali = dali_train = dali_val = []  # no need to load dali, files already exist
    # else:
    dali = get_dali()
    dali = dali[:500]  # use smaller dataset for testing
    print('Size of DALI:', len(dali))
    dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = DaliDataset(dali_train, 'train')
    val_data = DaliDataset(dali_val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate,)
    
    lyrics_database = LyricsDatabase(dali)

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    cfg = {'lr': config.lr,
           'batch_size': config.batch_size,
           'num_RCBs': config.num_RCBs,
           'channels': config.channels,
           'use_chars': config.use_chars,
           'context': config.context,
           'num_epochs': config.num_epochs,
           'dali_size': len(dali),
           'val_size': config.val_size,
           'num_negative_samples': config.num_negative_samples}

    print(cfg)

    config.time_report = TimeReport()

    config.time_report.add_timer('epoch')
    config.time_report.add_timer('train_loader')
    config.time_report.add_timer('batch.to(device)')
    config.time_report.add_timer('sampling negatives')
    config.time_report.add_timer('model')
    config.time_report.add_timer('loss')
    config.time_report.add_timer('backward')

    config.time_report.add_timer('audio_encoder')
    config.time_report.add_timer('audio_encoder.RCBs')
    config.time_report.add_timer('audio_encoder.conv1d')
    config.time_report.add_timer('text_encoder')
    config.time_report.add_timer('similarity')

    print('Training for one epoch')
    train(model, device, train_loader, lyrics_database, criterion, optimizer)
    
    config.time_report.report()

if __name__ == '__main__':
    print('Running time_models.py')
    main()
