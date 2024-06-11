# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from datetime import datetime

import config
from data import get_dali, get_jamendo, get_jamendoshorts, DaliDataset, NegativeSampler, collate
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
                for i in range(8):
                    heights = [positives_per_spectrogram[i], 50]
                    fig, axs = plt.subplots(2, 1, height_ratios=heights, figsize=(15, min((sum(heights) + 20 * len(heights)) // 12, 100)))

                    j, k = cumsum[i], cumsum[i + 1]
                    show(PA[j:k], axs[0], 'positive scores', positives[j:k])  # PA[i]
                    j = i * config.num_negative_samples
                    k = j + 50  # don't show all 1000 negative tokens
                    show(NA[j:k], axs[1], 'negative scores', negatives[j:k])  # NA[i]

                    fig.tight_layout()

                    wandb.log({'media/DALI_sample_' + str(i): plt, 'media/epoch': epoch})
                    plt.close()

    return val_loss / num_batches


from torch.utils.data import Dataset, DataLoader
from utils import words2phowords, wav2spec
import numpy as np
from models import AudioEncoder

class MyRandomDataset(Dataset):

    def __init__(self):
        self.words_list = [['this', 'is', 'me'], ['hello'], ['hi', "it's", 'me'], ['jen', 'sais', 'pas', 'de', 'tous'], ['last', 'one']] * 2
        self.phowords_list = [words2phowords(words) for words in self.words_list]
        self.spec_list = [wav2spec(np.random.randn(2222)) for _ in range(len(self.words_list))]

    def __getitem__(self, index):  # (spec, words, phowords)
        return (self.spec_list[index], self.words_list[index], self.phowords_list[index])

    def __len__(self):
        return len(self.words_list)
    
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

def main():
    fix_seeds()

    device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    batch_size = 4

    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    model = Model(input_size, output_size) #nn.DataParallel(AudioEncoder()) #SimilarityModel()
    model = nn.DataParallel(model)
    # display_module_parameters(model)
    #model = nn.DataParallel(model)
    model.to(device)

    negative_sampler = NegativeSampler([{'words': ['this', 'is', 'me']}, {'words': ['hello']}, {'words': ['hi', "it's", 'me']}, 
                                        {'words': ['jen', 'sais', 'pas', 'de', 'tous']}, {'words': ['last', 'one']}])

    #rand_loader = DataLoader(dataset=RandomDataset(), batch_size=batch_size, shuffle=False, collate_fn=collate)
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)
    for batch in rand_loader:
        input = batch.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
            "output_size", output.size())
        #spectrograms, positives, positives_per_spectrogram = batch
        #negatives = negative_sampler.sample(config.num_negative_samples, positives, positives_per_spectrogram)
        #negatives = torch.IntTensor(negatives)
        #spectrograms, positives, negatives = spectrograms.to(device), positives.to(device), negatives.to(device)
        #print("Outside: input size", spectrograms.shape, positives.shape, negatives.shape)
        #out = model(spectrograms)
        #PA, NA = model(spectrograms, positives, positives_per_spectrogram, negatives)
        #print("Outside: output_size", out.shape)

    return

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
           'val_size': config.val_size}
           #'dali_size': len(dali)}
    
    print(cfg)
    wandb.init(project='Train-Decode', config=cfg)

    run_start_time = datetime.now().strftime('%m-%d,%H:%M')
    run_checkpoint_dir = os.path.join(config.checkpoint_dir, run_start_time)
    os.makedirs(run_checkpoint_dir)

    device = torch.device('cuda:' + ','.join(config.gpu_ids) if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    model = SimilarityModel()
    # display_module_parameters(model)
    model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model.to(device)



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
    
    negative_sampler = NegativeSampler(dali)

    jamendo = get_jamendo()
    jamendoshorts = get_jamendoshorts()
    dali_val_20 = dali_val[:20]
    dali_train_20 = dali_train[:20]

    optimizer = optim.Adam(model.parameters(), config.lr)
    criterion = contrastive_loss

    # log before training
    epoch = -1

    #val_loss = validate(model, device, val_loader, negative_sampler, criterion, epoch)
    #wandb.log({'val/val_loss': val_loss, 'val/epoch': epoch})

    evaluate(model, device, jamendoshorts, log=True, epoch=epoch)
    PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo, log=False, epoch=-1)
    wandb.log({'metric/PCO_jamendo': PCO_jamendo, 'metric/epoch': epoch})
    wandb.log({'metric/AAE_jamendo': AAE_jamendo, 'metric/epoch': epoch})
    if not config.masked:
        PCO_dali_val_20, AAE_dali_val_20 = evaluate(model, device, dali_val_20, log=False, epoch=-1)
        PCO_dali_train_20, AAE_dali_train_20 = evaluate(model, device, dali_train_20, log=False, epoch=-1)
        wandb.log({'metric/PCO_dali_val_20': PCO_dali_val_20, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_dali_val_20': AAE_dali_val_20, 'metric/epoch': epoch})
        wandb.log({'metric/PCO_dali_train_20': PCO_dali_train_20, 'metric/epoch': epoch})
        wandb.log({'metric/AAE_dali_train_20': AAE_dali_train_20, 'metric/epoch': epoch})

    #model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '05-31,19:48', '6')))
    epoch = 0#7
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
            PCO_dali_val_20, AAE_dali_val_20 = evaluate(model, device, dali_val_20, log=False, epoch=-1)
            PCO_dali_train_20, AAE_dali_train_20 = evaluate(model, device, dali_train_20, log=False, epoch=-1)
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
