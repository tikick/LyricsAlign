# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import get_dali, LyricsAlignDataset, LyricsDatabase
from models import AudioEncoder, TextEncoder, SimilarityModel, data_processing
from test import validate
from utils import load_model, set_seed, count_parameters


def contrastive_loss(PA, NA):
    return torch.mean(torch.pow(torch.max(PA, dim=1).values - 1, 2) +
                      torch.mean(torch.pow(torch.max(NA, dim=1).values, 2)))  # max along time dimension


def train(model, device, train_loader, lyrics_database, criterion, optimizer):
    avg_time = 0.
    model.train()
    data_len = len(train_loader.dataset) // config.batch_size
    train_loss = 0.

    with tqdm(total=data_len) as pbar:
        for batch_idx, data in enumerate(train_loader):
            spectrograms, positives, len_positives = data
            spectrograms, positives = spectrograms.to(device), positives.to(device)

            t = time.time()

            optimizer.zero_grad()

            negatives = lyrics_database.sample(config.batch_size * config.num_negative_samples)
            negatives = torch.tensor(negatives, dtype=int).to(device)

            PA, NA = model(spectrograms, positives, len_positives, negatives)

            loss = criterion(PA, NA)
            loss.backward()

            optimizer.step()

            t = time.time() - t
            avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

            train_loss += loss.item()

            pbar.set_description('Current loss: {:.4f}'.format(loss))
            pbar.update(1)

    return train_loss / data_len


def main():
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # create folders for checkpoints and logs
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    # init models
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()
    model = SimilarityModel(audio_encoder, text_encoder).to(device)
    count_parameters(model)

    # prepare dataset
    if os.path.exists(os.path.join(config.pickle_dir, 'val.pkl')) and \
            os.path.exists(os.path.join(config.pickle_dir, 'train.pkl')) and \
            os.path.exists(os.path.join(config.pickle_dir, 'lyrics_database.pkl')):
        dali = dali_train = dali_val = None  # pickle files already saved
    else:
        dali = get_dali()
        dali_train, dali_val = train_test_split(dali, test_size=config.val_size, random_state=97)
    print('Size of DALI:', len(dali))

    train_data = LyricsAlignDataset(dali_train, 'train')
    val_data = LyricsAlignDataset(dali_val, 'val')

    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(dataset=train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=lambda x: data_processing(x),
                              **kwargs)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=lambda x: data_processing(x),
                            **kwargs)
    lyrics_database = LyricsDatabase(dali)

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    # training state dict for saving checkpoints
    state = {'num_worse_epochs': 0,  # number of consecutive epochs with loss >= best_loss
             'epoch_num': 0,
             'best_loss': np.Inf,
             'best_epoch': None}

    # load a pre-trained model
    if config.load_model is not None:
        state = load_model(model, config.load_model, device)

    from torch.utils.tensorboard import SummaryWriter
    import datetime
    current = datetime.datetime.now()
    writer = SummaryWriter(os.path.join(config.log_dir, current.strftime('%m:%d:%H:%M')))

    while state['num_worse_epochs'] < 20:  # stop after 20 epochs without improvement
        print('Epoch: ' + str(state['epoch_num']))

        # train
        train_loss = train(model, device, train_loader, lyrics_database, criterion, optimizer)
        print('Train loss: ' + str(train_loss))
        writer.add_scalar('train/loss', train_loss, state['epoch_num'])

        val_loss = validate(model, device, val_loader, lyrics_database, criterion)
        print('Val loss: ' + str(val_loss))
        writer.add_scalar('val/loss', val_loss, state['epoch_num'])

        if val_loss >= state['best_loss']:
            if state['epoch_num'] >= 20:  # after 20 epochs, start early stopping counts
                state['worse_epochs'] += 1
        else:
            print('MODEL IMPROVED ON VALIDATION SET!')
            state['num_worse_epochs'] = 0
            state['best_loss'] = val_loss
            state['best_epoch'] = state['epoch_num']

        # save checkpoint
        checkpoint_path = os.path.join(config.checkpoint_dir, 'checkpoint_' + str(state['epoch_num']))
        print('Saving model... best_epoch: {}, best_loss: {}'.format(state['best_epoch'], state['best_loss']))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'state': state
        }, checkpoint_path)

        state['epoch_num'] += 1

    writer.close()


if __name__ == '__main__':
    print('Running train.py')
    main()
