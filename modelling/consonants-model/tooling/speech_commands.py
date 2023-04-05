import pandas as pd
from dataset import Dataset
from dataset import DatasetStage

class SpeechCommandsDataset(Dataset):
    """
        See: https://www.tensorflow.org/datasets/catalog/speech_commands
    """
    NAME = 'speech_commands'
    DIR_PATTERN = '{sound_name}'
    DEFAULT_STAGE = DatasetStage.RAW
    DEFAULT_IS_LOCAL = True

    @staticmethod
    def extract_sound_name(row: pd.Series):
        return row['filepath'].split('/')[-2]

    @classmethod
    def extract_group(cls, row: pd.Series) -> str:
        return row['filepath'].split('/')[-1].split('_')[0]

    @staticmethod
    def extract_sample_name(row: pd.Series):
        return '/'.join(row['filepath'].split('/')[-2:]).replace('.wav', '')
