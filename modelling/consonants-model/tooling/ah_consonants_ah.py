import re
import pandas as pd
import numpy as np
import os
import sys

from dataset import Dataset, DatasetStage
from utils import load_wav


class AhConsonantsAhDataset(Dataset):
    NAME = 'ah_consonants_ah'
    DIR_PATTERN = '{group}'
    DEFAULT_STAGE = DatasetStage.TRIMMED
    DEFAULT_IS_LOCAL = False

    @staticmethod
    def extract_sound_name(row: pd.Series) -> str:
        sound_name = row['filename'].replace('.wav', '')
        sound_name = re.sub(r'\d+', '', sound_name).strip('-')
        sound_name = re.sub(r'ah', '', sound_name).strip('-')
        return sound_name

    @staticmethod
    def extract_sample_name(row: pd.Series) -> str:
        return '/'.join(row['filepath'].split('/')[-2:])

    @classmethod
    def extract_group(cls, row: pd.Series) -> str:
        return row['filepath'].split('/')[-2]
