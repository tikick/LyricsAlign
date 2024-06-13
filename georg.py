import config
import os
import pandas as pd

with open(os.path.join(config.georg_base, 'data', 'pytube_files.txt'), 'w') as f:
    for i in range(20):
        parq_file = os.path.join(config.georg_annotations, str(i), 'alignment.parq')
        df = pd.read_parquet(parq_file, engine='pyarrow')
        for _, row in df.iterrows():
                f.write(row['ytid'] + '.mp3\n')
