import pandas as pd

from dataset import Dataset, DatasetStage


class FsdNoisy18kDataset(Dataset):
    """
    See: http://www.eduardofonseca.net/FSDnoisy18k/
    """
    NAME = 'FSDnoisy18k'
    DIR_PATTERN = 'FSDnoisy18k.audio_train'
    DEFAULT_STAGE = DatasetStage.RAW
    DEFAULT_IS_LOCAL = True

    @staticmethod
    def extract_sample_name(row):
        return row['filepath'].split('/')[-1].replace('.wav', '')

    @classmethod
    def extract_group(cls, row: pd.Series) -> str:
        return f"FSDnoisy18k-{row['sample_name']}"

    @classmethod
    def extract_sound_name(cls, row: pd.Series) -> str:
        return 'negative'
