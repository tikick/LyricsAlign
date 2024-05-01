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
from data import get_dali, DaliDatasetPickle, LyricsDatabase, collate
from models import SimilarityModel, contrastive_loss
from utils import set_seed, count_parameters
from decode import align


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.
    batch_loss = 0.

    for idx, batch in enumerate(train_loader):
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

        if (idx + 1) % 100 == 0:
            train_metrics = {'train/batch_loss': batch_loss / 100, 'train/batch_idx': idx + 1}
            wandb.log({**train_metrics})
            batch_loss = 0.

    return train_loss / num_batches


def validate(model, device, val_loader, lyrics_database, criterion):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with torch.no_grad():
        for batch in val_loader:
            spectrograms, positives, len_positives = batch
            negatives = lyrics_database.sample(config.num_negative_samples, positives, len_positives)
            negatives = torch.IntTensor(negatives)
            spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

            PA, NA = model(spectrograms, positives, len_positives, negatives)

            loss = criterion(PA, NA)

            val_loss += loss.item()

    return val_loss / num_batches


def eval(model, device, eval_dataset, metric):
    model.eval()

    with torch.no_grad():
        for song in eval_dataset:
            spectrograms, positives, gt_alignment = song
            spectrograms, positives = spectrograms.to(device), positives.to(device)

            S = model(spectrograms, positives)
            S = S.cpu()  # detach?

            alignment = align(S)

            metric(alignment, gt_alignment)



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
    
    wandb.init(project='Lyrics Alignment', config=cfg)

    print(cfg)

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    best_loss = np.Inf

    # while num_worse_epochs < 20:  # stop after 20 epochs without improvement
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch)

        train_loss = train(model, device, train_loader, lyrics_database, criterion, optimizer)
        train_metrics = {'train/train_loss': train_loss, 'train/epoch': epoch}

        val_loss = validate(model, device, val_loader, lyrics_database, criterion)
        val_metrics = {'val/val_loss': val_loss, 'val/epoch': epoch}

        print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:3f}')
        
        wandb.log({**train_metrics, **val_metrics})

        if val_loss < best_loss:
            print('Model improved on validation set')
            best_loss = val_loss

            # save checkpoint
            checkpoint_path = os.path.join(config.checkpoint_dir, 'checkpoint_' + str(epoch))
            torch.save(model.state_dict(), checkpoint_path)

    wandb.finish()


if __name__ == '__main__':
    print('Running train.py')
    main()
