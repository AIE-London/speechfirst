import pandas as pd
from functools import lru_cache
import os

from dataset import Dataset, DatasetStage


class CommonVoiceDataset(Dataset):
    """
    See: https://commonvoice.mozilla.org/en
    """
    NAME = 'common_voice'
    DIR_PATTERN = 'en/wav'
    DEFAULT_STAGE = DatasetStage.RAW
    DEFAULT_IS_LOCAL = True

    @staticmethod
    def extract_sample_name(row):
        return row['filepath'].split('/')[-1].split('_')[-1].replace('.wav', '')

    @classmethod
    def extract_group(cls, row: pd.Series, stage=None) -> str:
        return cls.get_client_ids(stage=stage)[row['filename']]

    @staticmethod
    def extract_sound_name(row: pd.Series) -> str:
        return 'negative'

    @classmethod
    @lru_cache(maxsize=1)
    def get_client_ids(cls, stage=None):
        dataset_path = cls.get_path(stage=stage)
        client_ids = dict()
        # TODO:
        ref_files = ['dev.tsv', 'invalidated.tsv', 'other.tsv', 'test.tsv', 'train.tsv', 'validated.tsv']
        for ref_file in ref_files:
            ref_df = pd.read_csv(os.path.join(dataset_path, 'en', ref_file), sep='\t', header=0)[['client_id', 'path']]
            ref_df['path'] = ref_df['path'].apply(lambda x: x.replace('.mp3', '.wav'))
            ref_file_client_ids = pd.Series(ref_df['client_id'].values, index=ref_df['path']).to_dict()
            client_ids.update(ref_file_client_ids)
        return client_ids
