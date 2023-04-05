"""
TODO WIP: Integrating dataset class with kedro datasets. Barely started here and it is also outdated at this point.
"""
import hashlib
import os
import pathlib
from abc import ABC, abstractmethod
from glob import glob
from typing import Union

import boto3
import librosa
import s3fs
import sklearn
import pandas as pd
import numpy as np
from kedro.io import AbstractDataSet

SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()
SRC_PATH = pathlib.Path(SCRIPT_PATH).parent.absolute()
ROOT_PATH = pathlib.Path(SRC_PATH).parent.absolute()
DATA_PATH = LOCAL_DATA_PATH = os.path.join(ROOT_PATH, 'data')
S3_BUCKET = 'speech-first'


############
# Datasets #
############

class DatasetStage:
    RAW = 'raw'
    TRIMMED = 'trimmed'


class DatasetStageDir:
    RAW = 'data/01_raw'
    TRIMMED = 'data/02_intermediate/01_trimmed'


DEFAULT_STAGE = DatasetStage.RAW

DATASET_STAGE_DIR = {
    DatasetStage.RAW: DatasetStageDir.RAW,
    DatasetStage.TRIMMED: DatasetStageDir.TRIMMED
}


class Dataset(AbstractDataSet):
    NAME = None
    DEFAULT_STAGE = DatasetStage.RAW
    DIR_PATTERN = None
    DEFAULT_IS_LOCAL = False

    def __init__(self, local_filepath, s3_filepath=None, stage=DatasetStage.RAW, local=True, s3_credentials=None):
        self._local_filepath = local_filepath
        self._s3_filepath = s3_filepath
        self._local = local
        self._stage = stage
        if local is False:
            self._s3 = s3fs.S3FileSystem(**s3_credentials)
        self.update_files()

    def update_files(self):
        if self._local is True:
            self._files = glob(self._local_filepath, recursive=True)
        else:
            self._files = self._s3.glob(self._local_filepath)

    @classmethod
    def get_directory_pattern(cls):
        return cls.DIR_PATTERN

    @classmethod
    def get_path(cls, stage: str = None, local=None):
        """

        Args:
            stage: Stage of the dataset chosen from DatasetStage enum.
            local: True if local path, otherwise S3 path.
        Returns:

        """
        if stage is None:
            stage = cls.DEFAULT_STAGE
        if local is None:
            local = cls.DEFAULT_IS_LOCAL

        if local is True:
            root = ROOT_PATH
        else:
            root = ''
        return os.path.join(root, DATASET_STAGE_DIR[stage], cls.NAME)

    @classmethod
    def get_filepath(cls, filename: str, stage=None, local=True, **kwargs):
        dataset_path = cls.get_path(stage=stage, local=local)

        post_dataset_path = cls.get_directory_pattern()
        for kwarg, value in kwargs.items():
            post_dataset_path = post_dataset_path.replace(f'{{{kwarg}}}', value)
        if '{' in post_dataset_path:
            raise AttributeError(f'Some keywords have not been replaced with values in path: "{post_dataset_path}"')
        return os.path.join(dataset_path, post_dataset_path, filename)

    @staticmethod
    def _create_dummy_cols(df, target_col='sound_name', verbose=False):
        for target in df[target_col].unique():
            df[f'is_{target}'] = df[target_col] == target
        if verbose:
            print('Dummy columns created.')
        return df

    @staticmethod
    def _limit(all_files, limit=None, verbose=True):
        # Limit files if needed
        try:
            selected_files = all_files[:limit]
        except:
            selected_files = all_files

        if verbose:
            if all_files:
                print(f"Files selected: {len(selected_files)}/{len(all_files)} ({(len(selected_files) / len(all_files)) * 100:.2f}%)")
            else:
                print('No files found')
        return selected_files

    @staticmethod
    def _md5(filepath):
        """ Get MD5 hashsum of a file.

        Args:
            filepath: Path to the file.

        Returns:
            MD5 checksum.
        """
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @classmethod
    def save_locally(cls, filename: str, stage: str = None, s3_bucket: str = S3_BUCKET, check_md5=True,
                     verbose=True, ignore_errors=False, remove_local_if_no_s3=False, **kwargs) -> Union[str, None]:
        """Save a collected audio file locally

        Args:
            filename: Filename.
            stage: Dataset stage. If None, it will be the default stage of the dataset.
            s3_bucket: Bucket name in AWS S3.
            remove_local_if_no_s3: True, if local version should be removed.
            ignore_errors: True, if errors should be ignored rather than raised.
            check_md5: True, if MD5 checks should be performed for caching purposes.
            verbose: True if verbose output should be given.
            **kwargs: All variables needed for determining paths (e.g. group or sound_name)


        Returns:
            local filepath
        """
        s3 = boto3.client('s3')

        local_filepath = cls.get_filepath(filename, stage=stage, local=True, **kwargs)
        s3_filepath = cls.get_filepath(filename, stage=stage, local=False, **kwargs)

        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

        # Do not download if it exists (and has the same MD5 checksum)
        if os.path.exists(local_filepath):
            if check_md5:
                s3_md5 = s3.head_object(
                    Bucket=s3_bucket,
                    Key=s3_filepath
                )['ETag'][1:-1]
                local_md5 = cls._md5(local_filepath)
                if s3_md5 == local_md5:
                    # MD5 checksums are the same - all good
                    return local_filepath
                else:
                    if verbose:
                        print(f'{local_filepath} is different on S3')

            else:
                # No MD5 checking, we just assume it is all fine if file exists
                return local_filepath

        try:
            # Dealing with 0 byte S3 files, TODO this still leaves 0 byte files on S3
            s3_size = s3.head_object(
                Bucket=s3_bucket,
                Key=s3_filepath
            )['Size']
            if s3_size == 0:
                s3.delete_object(Bucket=s3_bucket, Key=s3_filepath)
                if verbose:
                    print(f'{s3_filepath} is 0 bytes. Deleted S3 file.')
                try:
                    os.remove(local_filepath)
                    if verbose:
                        print(f'{s3_filepath} is 0 bytes. Deleted local file.')
                except:
                    pass
                return None
            # Download file
            with open(local_filepath, 'wb') as f:
                s3.download_fileobj(s3_bucket, s3_filepath, f)
                if verbose:
                    print(f'Downloaded {s3_filepath} to {local_filepath}.')
        except:
            try:
                if remove_local_if_no_s3:
                    # Could not find S3 file, removing local file
                    if verbose:
                        print(f'{s3_filepath} does not exist. Removing local...')
                    os.remove(f)
            except:
                pass
            if ignore_errors:
                return None
            else:
                raise FileNotFoundError
        return local_filepath

    @classmethod
    @abstractmethod
    def extract_group(cls, row: pd.Series) -> str:
        return

    @staticmethod
    @abstractmethod
    def extract_sound_name(row: pd.Series) -> str:
        return

    @staticmethod
    def extract_sample_name(row: pd.Series) -> str:
        return row['filename'].replace('.wav', '')

    @staticmethod
    def load_wav(filepath: str, sample_rate: int = 16_000, ignore_errors: bool = True) -> np.array:
        '''Load the wav file as a 1-d numpy array.'''
        if not os.path.exists(filepath):
            if ignore_errors:
                return np.nan
            else:
                raise FileNotFoundError(filepath)
        try:
            data, sample_rate = librosa.load(filepath, sr=sample_rate)
        except Exception as e:
            if ignore_errors:
                return np.nan
            else:
                raise e

        return data

    @staticmethod
    def _lazy_load_wav_to_numbers(file):
        try:
            numbers = file().data[0]
            return np.array(numbers, dtype=np.float32)
        except:
            return None

    @classmethod
    def load(cls, context, limit=None, load_audio=True, create_dummy_cols=False,
             stage: str = None, local=True, shuffle=True, cache=True, check_md5=False,
             random_state=42, verbose=True) -> pd.DataFrame:
        """Load dataset

        Args:
            context: Kedro context.
            limit: Max number of files to retrieve.
            load_audio: True if audio should be loaded into 'audio' column (SR=16k)
            create_dummy_cols: True if dummy target columns should be created (i.e. is_{class})
            stage: DatasetStage
            local: True, if local files should be used rather than S3
            shuffle: True, if files should be shuffled (before limiting)
            cache: True, if S3 files should be cached locally.
            check_md5: True, if MD5 checks should be performed for caching.
            random_state: Random state.
            verbose: True, if verbose output should be printed.

        Returns:
            DataFrame with the dataset
        """

        if not stage:
            stage = cls.DEFAULT_STAGE

        all_files = cls.get_files(context, stage=stage)

        # Shuffle
        if shuffle:
            all_files = sklearn.utils.shuffle(all_files, random_state=random_state)

        # Limit
        selected_files = cls._limit(all_files, limit=limit, verbose=verbose)
        if not selected_files:
            return None

        if local is True:
            filepath_col = 'local_filepath'
        else:
            filepath_col = 's3_filepath'

        # Generate a dataframe from files on S3
        # df = pd.DataFrame({filepath_col: selected_files})
        df = pd.DataFrame()

        s3_dataset_path = cls.get_path(stage=stage, local=False)
        local_dataset_path = cls.get_path(stage=stage, local=True)

        if cls.DEFAULT_IS_LOCAL:
            df['local_filepath'] = selected_files
            df['s3_filepath'] = df['local_filepath'].apply(lambda x: x.replace(local_dataset_path, s3_dataset_path))
        else:
            df['s3_filepath'] = ['/'.join(f.split('/')[1:]) for f in selected_files]
            df['local_filepath'] = df['s3_filepath'].apply(lambda x: x.replace(s3_dataset_path, local_dataset_path))

        df['filepath'] = df[filepath_col]
        df['filename'] = df[filepath_col].apply(lambda x: x.split('/')[-1])
        df['sample_name'] = df.apply(cls.extract_sample_name, axis=1)
        df['sound_name'] = df.apply(cls.extract_sound_name, axis=1)
        df['group'] = df.apply(cls.extract_group, axis=1)

        # Create all the potential targets
        if create_dummy_cols:
            df = cls._create_dummy_cols(df, verbose=verbose)

        if local:
            if load_audio:
                df['audio'] = df['local_filepath'].apply(cls.load_wav)
        else:
            if cache:
                # Downloading files locally (caching)
                if verbose:
                    print(f'cache is enabled - using local files and downloading missing ones locally (MD5 checks are {"enabled" if check_md5 else "disabled"})...')
                df['local_filepath'] = df.apply(
                    lambda x: cls.save_locally(x.filename, check_md5=check_md5, ignore_errors=True, verbose=verbose, **x.to_dict()), axis=1)
                if load_audio:
                    print('load_audio is enabled. Loading audio into "audio" column...')
                    df['audio'] = df['local_filepath'].apply(cls.load_wav)
            elif cls.LAZY_CATALOG is not None:
                # No caching, get a new copy each time from 'file' column
                print('cache is disabled - preparing the lazy load on "file" column.')
                partial_file_load = context.catalog.load(cls.LAZY_CATALOG)
                df['file'] = df['sample_name'].apply(lambda x: partial_file_load[x])
                df['local_filepath'] = None
                if load_audio:
                    print('load_audio is enabled. Loading audio into "audio" column...')
                    df['audio'] = df['file'].apply(cls._lazy_load_wav_to_numbers)  # Does not use caching
            else:
                raise ValueError('Must have either cache enabled, or a lazy catalog specified.')
            print('Done.')
        return df
