import config
import os

with open(os.path.join(config.georg_base, 'data', 'audio_ids.txt'), 'r') as r:
    with open(os.path.join(config.georg_base, 'data', 'vocals_ids.txt'), 'w') as w:
        for line in r.readlines():
            w.write(line[:-5] + '_vocals.mp3\n')
