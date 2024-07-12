# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import os
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from media import show_plot


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    num_batches = len(train_loader.dataset) // config.batch_size
    train_loss = 0.
    batch_loss = 0.

    for idx, batch in enumerate(tqdm(train_loader)):
        spectrograms, positives, times, is_duplicate, positives_per_spectrogram = batch
        spectrograms, positives = spectrograms.to(device), positives.to(device)

        optimizer.zero_grad()

        PA = model(spectrograms, positives, positives_per_spectrogram)

        loss = criterion(PA, times, is_duplicate)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        batch_loss += loss.item()

        log_freq = 100  # log every 100 batches
        if (idx + 1) % log_freq == 0:
            wandb.log({'train/batch_loss': batch_loss / log_freq, 'train/batch_idx': idx + 1})
            batch_loss = 0.

    return train_loss / num_batches


def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    num_batches = len(val_loader.dataset) // config.batch_size
    val_loss = 0.

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            spectrograms, positives, times, is_duplicate, positives_per_spectrogram = batch
            spectrograms, positives = spectrograms.to(device), positives.to(device)

            PA = model(spectrograms, positives, positives_per_spectrogram)

            loss = criterion(PA, times, is_duplicate)
            val_loss += loss.item()

            # log first batch
            if idx == 0:
                PA = PA.cpu().numpy()
                positives = positives.cpu().tolist()

                f = int2char if config.use_chars else int2phoneme
                positives = [[f[pos[i]] for i in range(len(pos))] for pos in positives]

                cumsum = np.cumsum([0] + positives_per_spectrogram)
                for i in range(min(8, config.batch_size)):
                    heights = [positives_per_spectrogram[i]]
                    fig, ax = plt.subplots(figsize=(15, min((sum(heights) + 20 * len(heights)) // 12, 100)))

                    j, k = cumsum[i], cumsum[i + 1]
                    show_plot(PA[j:k], ax, 'positive scores', positives[j:k])  # PA[i]

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
           'alpha': config.alpha,
           'box_slack': config.box_slack,
           'loss': config.loss,
           'masked': config.masked,
           'use_dali': config.use_dali,
           'val_size': config.val_size}
    
    print(cfg)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project='New-Align', config=cfg)

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
    
    #negative_sampler = NegativeSampler(dataset)

    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()
    train_20 = train_split[:20]
    val_20 = val_split[:20]

    optimizer = optim.Adam(model.parameters(), config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold=1e-3, threshold_mode='abs')

    criterion = contrastive_loss

    epoch = -1
    #model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '07-05,10:48', str(epoch))))
    epoch += 1
    while epoch < config.num_epochs:
        print('Epoch:', epoch)

        train_loss = train(model, device, train_loader, criterion, optimizer)
        wandb.log({'train/train_loss': train_loss, 'train/epoch': epoch})

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, str(epoch)))

        evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
        PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False)
        wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
        if not config.masked:
            PCO_val_20, AAE_val_20 = evaluate(model, device, val_20, log=False)
            PCO_train_20, AAE_train_20 = evaluate(model, device, train_20, log=False)
            wandb.log({'metric/PCO_val_20': PCO_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_val_20': AAE_val_20, 'metric/epoch': epoch})
            wandb.log({'metric/PCO_train_20': PCO_train_20, 'metric/epoch': epoch})
            wandb.log({'metric/AAE_train_20': AAE_train_20, 'metric/epoch': epoch})

        val_loss = validate(model, device, val_loader, criterion, epoch)
        wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})
        
        old_lr = optimizer.param_groups[0]["lr"]
        #scheduler.step(PCO_val_20)  # error if masked = True
        scheduler.step(PCO_jamendo)
        new_lr = optimizer.param_groups[0]["lr"]
        print(f'lr: {old_lr} -> {new_lr}')

        print(f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:3f}, PCO: {PCO_jamendo}, AAE: {AAE_jamendo}')

        epoch += 1

    wandb.finish()


if __name__ == '__main__':
    print('Running train.py')
    main()
