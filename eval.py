import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

import config
from data import get_dali, DaliDataset, LyricsDatabase, collate, jamendo_collate
from models import SimilarityModel, contrastive_loss
from utils import set_seed, count_parameters
from decode import align


def evaluate(model, device, jamendo, metric):
    model.eval()

    with torch.no_grad():
        for song in jamendo:
            spectrogram, positives = jamendo_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu()  # detach?

            alignment = align(S, song)

            metric(alignment, gt_alignment)
