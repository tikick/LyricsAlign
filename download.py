import DALI as dali_code
import os
from pytube import YouTube

import config

dali_data = dali_code.get_the_DALI_dataset(config.dali_annot_path, skip=[], keep=[])

base_url = 'https://www.youtube.com/watch?v='
lang = 'english'

if not os.path.isdir(config.dali_audio_path):
    os.makedirs(config.dali_audio_path)

annot_list = os.listdir(config.dali_annot_path)
for file in annot_list:
    dali_id = file[:-3]

    if lang is not None and dali_data[dali_id].info['metadata']['language'] != lang:
        continue

    url = base_url + dali_data[dali_id].info['audio']['url']

    try:
        yt = YouTube(url)

        # extract only audio 
        audio = yt.streams.filter(only_audio=True).first()

        # download the file 
        out_file = audio.download(output_path=config.dali_audio_path)

        new_file = os.path.join(config.dali_audio_path, dali_id + '.wav')
        os.rename(out_file, new_file)

        print(yt.title + ' has been successfully downloaded.')
    except:
        print('Failed to download ' + dali_data[dali_id].info['title'])
