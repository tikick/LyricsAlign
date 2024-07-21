import os
import torch
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from data import get_jamendo, get_jamendoshorts, jamendo_collate, get_dali, get_non_monotonic_dali
from models import SimilarityModel
from utils import fix_seeds
from decode import get_alignment
import example_decoding
import json


def evaluate(model, device, eval_dataset, file_name):
    model.eval()
    PCO_score_sum = 0.
    AAE_score_sum = 0.
    #dali_08_alignments = []  # dali songs with (0.3 <=) PCO < 0.8
    #dali_08_ids = []
    #dali_08_wrong_words = []
    #dali_03_alignments = []  # PCO < 0.3
    #dali_03_ids = []
    #dali_03_wrong_words = []
    #jamendo_wrong_words = []
    non_monotonic_dali_words = []

    #file_path = os.path.join(config.base_path, file_name)
    #with open(file_path, 'r') as f:
    #    for line in f:
    #        # Read each line (which contains a JSON-encoded list)
    #        word_alignment = json.loads(line.strip())  # Convert JSON string to list
    #        word_alignments.append(word_alignment)

    with torch.no_grad():
        for song in tqdm(eval_dataset):
            spectrogram, positives = jamendo_collate(song)
            spectrogram, positives = spectrogram.to(device), positives.to(device)

            S = model(spectrogram, positives)
            S = S.cpu().numpy()

            _, word_alignment = get_alignment(S, song, time_measure='seconds')
        
            PCO_score, all_words = percentage_of_correct_onsets(song['words'], word_alignment, song['times'])
            AAE_score = average_absolute_error(word_alignment, song['times'])

            #if PCO_score < 0.3:
                #dali_03_alignments.append(word_alignment)
                #dali_03_ids.append(song['id'])
            #    dali_03_wrong_words.append([f'id: {song['id']}', f'url: {song['url']}', ''])
            #    dali_03_wrong_words.append([f'PCO: {PCO_score:.4f}', f'AAE: {AAE_score:.4f}', ''])
            #    dali_03_wrong_words.append(['', '', ''])
            #    dali_03_wrong_words.append(['gt_time', 'time_dif', 'word'])
            #    dali_03_wrong_words += wrong_words
            #    dali_03_wrong_words.append(['\n', '\n', '\n'])
            #elif PCO_score < 0.8:
                #dali_08_alignments.append(word_alignment)
                #dali_08_ids.append(song['id'])
            #    dali_08_wrong_words.append([f'id: {song['id']}', f'url: {song['url']}', ''])
            #    dali_08_wrong_words.append([f'PCO: {PCO_score:.4f}', f'AAE: {AAE_score:.4f}', ''])
            #    dali_08_wrong_words.append(['', '', ''])
            #    dali_08_wrong_words.append(['gt_time', 'time_dif', 'word'])
            #    dali_08_wrong_words += wrong_words
            #    dali_08_wrong_words.append(['\n', '\n', '\n'])

            non_monotonic_dali_words.append([f'id: {song['id']}', f'url: {song['url']}', ''])
            non_monotonic_dali_words.append([f'PCO: {PCO_score:.4f}', f'AAE: {AAE_score:.4f}', ''])
            non_monotonic_dali_words.append(['', '', ''])
            non_monotonic_dali_words.append(['gt_time', 'time_dif', 'word'])
            non_monotonic_dali_words += all_words
            non_monotonic_dali_words.append(['\n', '\n', '\n'])
                
            print(f'non_monotonic_dali_id: {song['id']}, PCO: {PCO_score:.4f}, AAE: {AAE_score:.4f}')

            PCO_score_sum += PCO_score
            AAE_score_sum += AAE_score
    
    #file_path = os.path.join(config.base_path, 'dali_03_alignments.txt')
    #with open(file_path, 'w') as f:
    #    for word_alignment in dali_03_alignments:
    #        json.dump(word_alignment, f)  # Write list as JSON string
    #        f.write('\n')  # Add a newline after each list

    #file_path = os.path.join(config.base_path, 'dali_03_ids.txt')
    #with open(file_path, 'w') as f:
    #    for id in dali_03_ids:
    #        f.write(id + '\n')

    file_path = os.path.join(config.base_path, file_name)
    with open(file_path, 'w') as f:
        for s in non_monotonic_dali_words:
            f.write('{: <20} {: <20} {: <20}\n'.format(*s))

    return PCO_score_sum / len(eval_dataset), AAE_score_sum / len(eval_dataset)


def average_absolute_error(alignment, gt_alignment):
    assert len(alignment) == len(gt_alignment)
    abs_error = 0.
    for time, gt_time in zip(alignment, gt_alignment):
        abs_error += abs(time[0] - gt_time[0])
    return abs_error / len(alignment)


def percentage_of_correct_onsets(words, alignment, gt_alignment, tol=0.3):
    all_words = []
    assert len(alignment) == len(gt_alignment) and len(words) == len(alignment)
    correct_onsets = 0
    for idx, (word, time, gt_time) in enumerate(zip(words, alignment, gt_alignment)):
        if abs(time[0] - gt_time[0]) <= tol:
            correct_onsets += 1
    
        dif = time[0] - gt_time[0]
        all_words.append([f'{gt_time[0]:.2f}', f'{dif:.2f}', f'{word}'])

    return correct_onsets / len(alignment), all_words


if __name__ == '__main__':
    print('Running stats.py')

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
    #os.environ["WANDB__SERVICE_WAIT"] = "300"
    #wandb.init(project='New-Align', config=cfg)

    #device = torch.device('cuda')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = SimilarityModel().to(device)
    #jamendo = get_jamendo()
    #jamendoshorts = get_jamendoshorts()

    #model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, '07-05,10:48', str(3))))

    #train_20 = get_dali(keep=['68b6083c1f5e4d6ea296a89c0a774da2', 'e171945c761749f0a157694c57213271', '9c7747023dbc4b94bebceb3365a69be5',
    #                         '892876b7ef924a9baac482caa00c674e', 'e0ed5513279d4e759c960a0ae098335e', 'b252929f8f284a08afc91ba06d643deb',
    #                         'cfaaa5e9ab1c42bd8e1f00b149a9fe1c', 'e646ad4710d64f178013d0eb8e404369', 'bcdc1568db8a4e298e4e1f0651475725',
    #                         '373dc6adcac342f2b9ee9c22ba831112', '10709a6de99d4668a05e91d2df588ca9', '663f04fe63cd45dfb4f96c6fc0846022',
    #                         '43a3ee52120b448abdea76d8f277eb3b', '3559797f9a1647209578fcfbfa851e47', '8d319219cdc94e10ae5aaf8ecfdbfd2a',
    #                         '8dc5d778a80c4c759ac433679e5a00be', 'eb3c0842190b42e29b1df91a55a3c7c8', '2fe0fcc9713947f195821d778d7a9f12',
    #                         '4688353ef25e43dbb256490a0272bfeb', '79fe3f63bbd044878309c10c520344d6'])
    sorted_non_monotonic_dali, old_non_monotonic_dali = get_non_monotonic_dali()
    #dali = dali[:20]
    #print('Size of DALI:', len(dali))
    #train_split, val_split = train_test_split(dataset, test_size=config.val_size, random_state=97)
    #train_20 = train_split[:20]
    #val_20 = val_split[:20]
    #val_20 = get_dali(keep=['bf08fac2f9c142a9adb494435aca7b05', 'b2128361a7f04cb8a1f8361b36c03473', 'a8e300f064d746a794b2ccc92aa73d91',
    #                        '9ea84af95ce54636b956b0dfb40a6905', '5929748602c34694868c67c041c54eb6', '65d3f2c0da1e49989197ec8d8eeff551',
    #                        '018f045bb1784602976a289705832d60', '9c61fe7a08744655900996e56146032d', '103c2c259dc74b899aed8b60f5fd9199',
    #                        '672779cb487f4631b4c08570cc047c1f', '2db82968ee5847b0bbd51b855ba82905', 'db5f42997eb5424c8021c3106e86b09e',
    #                        '914882a4c3564bf087d957ed1983c70b', 'e07c5adf6347453c98e40bb0f94cb821', '1a03cd6eb6df4970b1ac7f72b4295dac',
    #                        '28162a764a0c4c76a4262f33f8a3a3aa', 'abcd378dc6f8465e90bad85c69081dd4', '532ddac0de1b4f78a014a31b2585266f',
    #                        'a0c086f5bb064159aa609cd38b68e92f', '275f6a8de8b94f52a4147e1c2cad691d'])

    #print('train_20')
    #PCO_train_20, AAE_train_20 = evaluate(None, None, train_20, "dali_train_20_alignments.txt")
    #print(f'PCO_train_20: {PCO_train_20}, AAE_train_20: {AAE_train_20}')

    #print('val_20')
    #PCO_val_20, AAE_val_20 = evaluate(None, None, val_20, "dali_val_20_alignments.txt")
    #print(f'PCO_val_20: {PCO_val_20}, AAE_val_20: {AAE_val_20}')

    #PCO_sorted_non_monotonic_dali, AAE_sorted_non_monotonic_dali = evaluate(model, device, sorted_non_monotonic_dali, 'sorted_non_monotonic_dali_words.txt')
    #print(f'PCO_sorted_non_monotonic_dali: {PCO_sorted_non_monotonic_dali}, AAE_sorted_non_monotonic_dali: {AAE_sorted_non_monotonic_dali}')

    #PCO_old_non_monotonic_dali, AAE_old_non_monotonic_dali = evaluate(model, device, old_non_monotonic_dali, 'old_non_monotonic_dali_words.txt')
    #print(f'PCO_old_non_monotonic_dali: {PCO_old_non_monotonic_dali}, AAE_old_non_monotonic_dali: {AAE_old_non_monotonic_dali}')

    #PCO_jamendo, AAE_jamendo = evaluate(model, device, jamendo)
    #print(f'PCO_jamendo: {PCO_jamendo}, AAE_jamendo: {AAE_jamendo}')