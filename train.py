# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime

import config
from data import get_dali, get_georg, get_jamendo, get_jamendoshorts, LA_Dataset, NegativeSampler, collate
from models import SimilarityModel, contrastive_loss
from utils import fix_seeds, display_module_parameters, int2char, int2phoneme
from eval import evaluate
from decode import show


def train(model, device, train_loader, negative_sampler, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.
    batch_loss = 0.

    for idx, batch in enumerate(tqdm(train_loader)):
        spectrograms, positives, positives_per_spectrogram = batch
        negatives = negative_sampler.sample(config.num_negative_samples, positives, positives_per_spectrogram)
        negatives = torch.IntTensor(negatives)
        spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

        optimizer.zero_grad()

        PA, NA = model(spectrograms, positives, positives_per_spectrogram, negatives)

        loss = criterion(PA, NA)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        batch_loss += loss.item()

        log_freq = 100  # log every 100 batches
        if (idx + 1) % log_freq == 0:
            wandb.log({'train/batch_loss': batch_loss / log_freq, 'train/batch_idx': idx + 1})
            batch_loss = 0.

    return train_loss / num_batches


def validate(model, device, val_loader, negative_sampler, criterion, epoch):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            spectrograms, positives, positives_per_spectrogram = batch
            negatives = negative_sampler.sample(config.num_negative_samples, positives, positives_per_spectrogram)
            negatives = torch.IntTensor(negatives)
            spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)

            PA, NA = model(spectrograms, positives, positives_per_spectrogram, negatives)

            loss = criterion(PA, NA)
            val_loss += loss.item()

            # log first batch
            if idx == 0:
                PA = PA.cpu().numpy()
                NA = NA.cpu().numpy()
                #PA = 0.5 * (PA + 1)
                #NA = 0.5 * (NA + 1)
                positives = positives.cpu().tolist()
                negatives = negatives.cpu().tolist()

                f = int2char if config.use_chars else int2phoneme
                positives = [[f[pos[i]] for i in range(len(pos))] for pos in positives]
                negatives = [[f[neg[i]] for i in range(len(neg))] for neg in negatives]

                cumsum = np.cumsum([0] + positives_per_spectrogram)
                for i in range(min(8, config.batch_size)):
                    heights = [positives_per_spectrogram[i], 50]
                    fig, axs = plt.subplots(2, 1, height_ratios=heights, figsize=(15, min((sum(heights) + 20 * len(heights)) // 12, 100)))

                    j, k = cumsum[i], cumsum[i + 1]
                    show(PA[j:k], axs[0], 'positive scores', positives[j:k])  # PA[i]
                    j = i * config.num_negative_samples
                    k = j + 50  # don't show all 1000 negative tokens
                    show(NA[j:k], axs[1], 'negative scores', negatives[j:k])  # NA[i]

                    fig.tight_layout()

                    wandb.log({'media/val_sample_' + str(i): plt, 'media/epoch': epoch})
                    plt.close()

    return val_loss / num_batches


def main():
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
           'masked': config.masked,
           'use_dali': config.use_dali,
           'words_slack': config.words_slack,
           'val_size': config.val_size}
    
    print(cfg)
    wandb.init(project='Train-Decode', config=cfg)

    run_start_time = datetime.now().strftime('%m-%d,%H:%M')
    run_checkpoint_dir = os.path.join(config.checkpoint_dir, run_start_time)
    os.makedirs(run_checkpoint_dir)

    device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = SimilarityModel().to(device)
    # display_module_parameters(model)

    # if train and val files already exist:
        # dali = dali_train = dali_val = []  # no need to load dali, files already exist
    # else:
    if config.use_dali:
        dataset = get_dali()
        print('Size of DALI:', len(dataset))
    else:
        dataset = get_georg()
        print('Size of Georg:', len(dataset))
    train_split, val_split = train_test_split(dataset, test_size=config.val_size, random_state=97)

    train_data = LA_Dataset(train_split, 'train')
    val_data = LA_Dataset(val_split, 'val')
    print('Num training samples:', len(train_data))
    print('Num validation samples:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    negative_sampler = NegativeSampler(dataset)

    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()
    train_20 = train_split[:20]
    val_20 = val_split[:20]

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    # log before training
    epoch = -1

    #val_loss = validate(model, device, val_loader, negative_sampler, criterion, epoch)
    #wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

    if False:
        evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)
        wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
        if not config.masked:
            PCO_val_20, AAE_val_20 = evaluate(model, device, val_20, log=False, epoch=-1)
            PCO_train_20, AAE_train_20 = evaluate(model, device, train_20, log=False, epoch=-1)
            wandb.log({'metric/PCO_val_20': PCO_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_val_20': AAE_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/PCO_train_20': PCO_train_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_train_20': AAE_train_20, 'metric/epoch': epoch})

    epoch = 1
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '06-15,12:08', str(epoch))))
    epoch += 1
    while epoch < config.num_epochs:
        print('Epoch:', epoch)

        train_loss = train(model, device, train_loader, negative_sampler, criterion, optimizer)
        wandb.log({'train/train_loss': train_loss, 'train/epoch': epoch})

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, str(epoch)))

        val_loss = validate(model, device, val_loader, negative_sampler, criterion, epoch)
        wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

        evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)
        wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
        if not config.masked:
            PCO_val_20, AAE_val_20 = evaluate(model, device, val_20, log=False, epoch=-1)
            PCO_train_20, AAE_train_20 = evaluate(model, device, train_20, log=False, epoch=-1)
            wandb.log({'metric/PCO_val_20': PCO_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_val_20': AAE_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/PCO_train_20': PCO_train_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_train_20': AAE_train_20, 'metric/epoch': epoch})

        print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:3f}, PCO: {PCO_jamendo}, AAE: {AAE_jamendo}')

        epoch += 1

    wandb.finish()


if __name__ == '__main__':
    print('Running train.py')
    main()
