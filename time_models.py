# From https://github.com/NVlabs/DiffRL/blob/main/utils/time_report.py

# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.
    
    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(self.name)
        self.start_time = time.time()
        
    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(self.name)
        self.time_total += time.time() - self.start_time
        self.start_time = None
    
    def report(self):
        print_info('Time report [{}]: {:.2f} seconds'.format(self.name, self.time_total))

    def clear(self):
        self.start_time = None
        self.time_total = 0.

class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name = name)
    
    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()
    
    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()
    
    def report(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for timer_name in self.timers.keys():
                self.timers[timer_name].report()
            print_info("-----------------------------------")
    
    def clear_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()
    
    def pop_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}




# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#import wandb

import config
from data import get_dali, DaliDatasetPickle, LyricsDatabase, collate
from models import AudioEncoder, TextEncoder, TimeSimilarityModel
from utils import set_seed, count_parameters


def contrastive_loss(PA, NA):
    return torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
           torch.mean(torch.pow(torch.max(NA, dim=1).values, 2))  # max along time dimension


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    #num_batches = 1

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

    train_loader = iter(train_loader)
    with tqdm(total=num_batches) as pbar:

        config.time_report.start_timer('epoch')

        for _ in range(num_batches):

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
            
            pbar.update(1)

        torch.cuda.synchronize()
        config.time_report.end_timer('epoch')

    config.time_report.report()

def main():
    config.time_report = TimeReport()

    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = TimeSimilarityModel().to(device)

    # if train and val files already exist:
        # dali = dali_train = dali_val = []  # no need to load dali, files already exist
    # else:
    dali = get_dali()
    dali = dali[:500]  # use smaller dataset for testing
    print('Size of DALI:', len(dali))
    dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)

    train_data = DaliDatasetPickle(dali_train, 'train')
    val_data = DaliDatasetPickle(dali_val, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=config.num_workers if device.type == 'cuda' else 0)
    val_loader = DataLoader(  dataset=val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate,
                              num_workers=config.num_workers if device.type == 'cuda' else 0)
    
    lyrics_database = LyricsDatabase(dali)

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    for epoch in range(1):
        print('Epoch:', epoch)
        train(model, device, train_loader, lyrics_database, criterion, optimizer)


if __name__ == '__main__':
    print('Running time_models.py')
    main()
