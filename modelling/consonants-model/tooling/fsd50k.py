import pandas as pd
import os
import json
from functools import lru_cache

from dataset import Dataset, DatasetStage


class Fsd50kDataset(Dataset):
    """
    See: https://zenodo.org/record/4060432
    """
    NAME = 'FSD50K'
    DIR_PATTERN = 'FSD50K.dev_audio'
    DEFAULT_STAGE = DatasetStage.RAW
    DEFAULT_IS_LOCAL = True

    @staticmethod
    def extract_sample_name(row):
        return row['filepath'].split('/')[-1].replace('.wav', '')

    @classmethod
    def extract_group(cls, row: pd.Series, stage=None) -> str:
        return cls.get_authors(stage=stage)[row['sample_name']]

    @staticmethod
    def extract_sound_name(row: pd.Series) -> str:
        return 'negative'

    @classmethod
    @lru_cache(maxsize=1)
    def get_authors(cls, stage=None):
        # TODO: use the test set too, not only dev
        dataset_path = cls.get_path(stage=stage)
        authors = dict()
        with open(os.path.join(dataset_path, 'FSD50K.metadata', 'dev_clips_info_FSD50K.json')) as f:
            metadata = json.load(f)
        for sample_name in metadata:
            authors[sample_name] = metadata[sample_name]['uploader']
        return authors
