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
from models import AudioEncoder, TextEncoder, SimilarityModel
from utils import set_seed, count_parameters


def contrastive_loss(PA, NA):
    return torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2)) + \
           torch.mean(torch.pow(torch.max(NA, dim=1).values, 2))  # max along time dimension


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.

    with tqdm(total=num_batches) as pbar:
        for idx, batch in enumerate(train_loader):
            spectrograms, positives, len_positives = batch
            spectrograms, positives = spectrograms.to(device), positives.to(device)

            optimizer.zero_grad()

            negatives = lyrics_database.sample(config.batch_size * config.num_negative_samples)
            negatives = torch.IntTensor(negatives).to(device)

            PA, NA = model(spectrograms, positives, len_positives, negatives)

            loss = criterion(PA, NA)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            pbar.set_description('Current loss: {:.4f}'.format(loss))
            pbar.update(1)

            if idx % 100 == 0:
                train_metrics = {'train/train_loss': train_loss, 'train/batch_idx': idx}
                wandb.log({**train_metrics})

    return train_loss / num_batches


def validate(model, device, val_loader, lyrics_database, criterion):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with tqdm(total=num_batches) as pbar:
        for batch in val_loader:
            spectrograms, positives, len_positives = batch
            spectrograms, positives = spectrograms.to(device), positives.to(device)

            negatives = lyrics_database.sample(config.batch_size * config.num_negative_samples)
            negatives = torch.IntTensor(negatives).to(device)

            PA, NA = model(spectrograms, positives, len_positives, negatives)

            loss = criterion(PA, NA)

            val_loss += loss.item()

            pbar.set_description("Current loss: {:.4f}".format(loss))
            pbar.update(1)

    return val_loss / num_batches


def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()
    model = SimilarityModel(audio_encoder, text_encoder).to(device)
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

    wandb.init(
        project='Lyrics Alignment',
        config={
            'lr': config.lr,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
        }
    )

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
        
        wandb.log({**train_metrics, **val_metrics})

        print(f'Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}')

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
