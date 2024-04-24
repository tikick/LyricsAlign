import DALI as dali_code
import os
from pytube import YouTube

import config

dali_data = dali_code.get_the_DALI_dataset(config.dali_annot_path, skip=[], keep=[])

base_url = 'https://www.youtube.com/watch?v='
lang = 'english'

if not os.path.isdir(config.dali_audio_path):
    os.makedirs(config.dali_audio_path)

num_downloads = 0
num_fails = 0
fails_url = []
fails_dali_id = []

annot_list = os.listdir(config.dali_annot_path)
for file in annot_list:
    dali_id = file[:-3]

    if lang is not None and dali_data[dali_id].info['metadata']['language'] != lang:
        continue

    url = base_url + dali_data[dali_id].info['audio']['url']

    try:
        video = YouTube(url)
        stream = video.streams.filter(only_audio=True).first()
        stream.download(output_path=config.dali_audio_path, filename=dali_id + '.wav')
        num_downloads += 1
    except Exception as e:
        num_fails += 1
        print(f'Failed to download dali_id={dali_id}, url={url}: {repr(e)}')

print(f'Successfully donwloaded {num_downloads} / {num_downloads + num_fails} songs')