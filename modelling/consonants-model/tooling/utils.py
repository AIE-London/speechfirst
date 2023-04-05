"""This module contains various utility functions for working with audio, S3, etc."""
import hashlib

import pandas as pd
import numpy as np

from scipy.io import wavfile
import librosa

import os
import re  # Just habit
from io import BytesIO
import base64

from glob import glob
from IPython.display import display, HTML, Audio

# Visualization 
import matplotlib.pyplot as plt

import scipy.io.wavfile

# AWS
import s3fs
import boto3

import pathlib

# For easy retrieval of various paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()
SRC_PATH = pathlib.Path(SCRIPT_PATH).parent.absolute()
ROOT_PATH = pathlib.Path(SRC_PATH).parent.absolute()
DATA_PATH = LOCAL_DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODEL_PATH = os.path.join(ROOT_PATH, 'data', '06_models')
CONFIG_PATH = os.path.join(ROOT_PATH, 'conf')
S3_BUCKET = 'speech-first'


#############################
#  Working with WAV files   #
#############################

def wavfile_to_html(filepath: str) -> str:
    """Converts WAV to an HTML component

    Args:
        filepath: WAV Filepath.

    Returns:
        HTML code
    """
    if isinstance(filepath, str) and filepath.endswith('.wav'):
        return f'''<audio src="{filepath}" controls ></audio>'''
    else:
        return filepath


def display_dataframe(df: pd.DataFrame) -> HTML:
    '''Replaces strings ending in .wav with a playable clip.'''

    original_col_width = pd.options.display.max_colwidth
    pd.set_option('display.max_colwidth', None)

    visual_dataframe = HTML(df.to_html(escape=False))

    pd.set_option('display.max_colwidth', original_col_width)
    return visual_dataframe


def save_wav(filepath: str, data: np.ndarray, sample_rate: int = 16_000):
    '''Load the wav file as a 1-d numpy array.'''
    if len(data) == 0:
        print(f"Skipping '{filepath}' because it's empty")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    scipy.io.wavfile.write(filepath, sample_rate, data)


def load_wav(filepath: str, sample_rate: int = 16_000, ignore_errors: bool = True) -> np.array:
    '''Load the wav file as a 1-d numpy array.'''

    try:
        data, sample_rate = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        if ignore_errors:
            data = np.nan
        else:
            raise e

    return data

read_wav = load_wav  # alias

def make_playable_wav(file):
    try:
        if type(file) == str:
            data = load_wav(file)
        elif isinstance(file, np.ndarray):
            data = file
        elif isinstance(file, pd.DataFrame):
            data = file().data[0]
        html = Audio(data=data, rate=16_000)._repr_html_().strip()
        return html
    except:
        return "Deleted"


def make_sparkline(data,
                   point=True, point_color='red', point_marker='.',
                   point_fill='red', point_size=6, point_alpha=1.0,
                   fill=True, fill_color='blue', fill_alpha=0.1,
                   figsize=(4, 0.25), xlim=None, ylim=None, **kwargs):
    """Create a single HTML image tag containing a base64 encoded
    sparkline style plot

    Parameters
    ----------
    data : array-like (list, 1d Numpy array, Pandas Series) sequence of
        data to plot
    point : bool, show point marker on last point on right
    point_location : not implemented, always plots rightmost
    point_color : str, matplotlib color code for point, default 'red'
    point_marker : str, matplotlib marker code for point
    point_fill : str, matplotlib marker fill color for point, default 'red'
    point_size : int, matplotlib markersize, default 6
    point_alpha : float, matplotlib alpha transparency for point
    fill : bool, show fill below line
    fill_color : str, matplotlib color code for fill
    fill_alpha : float, matplotlib alpha transparency for fill
    figsize : tuple of float, length and height of sparkline plot.  Attribute
        of matplotlib.pyplot.plot.
    **kwargs : keyword arguments passed to matplotlib.pyplot.plot
    """

    data = list(data)

    fig = plt.figure(figsize=figsize)  # set figure size to be small
    ax = fig.add_subplot(111)
    plot_len = len(data)
    plot_min = min(data)
    point_x = plot_len - 1

    plt.plot(data, **kwargs)

    # turn off all axis annotations    
    ax.axis('off')

    # fill between the axes
    plt.fill_between(range(plot_len), data, plot_len * [plot_min],
                     color=fill_color, alpha=fill_alpha)

    # plot the right-most point red, probably on makes sense in timeseries
    plt.plot(point_x, data[point_x], color=point_fill,
             marker=point_marker, markeredgecolor=point_color,
             markersize=point_size,
             alpha=point_alpha, clip_on=False)

    # squeeze axis to the edges of the figure
    fig.subplots_adjust(left=0)
    fig.subplots_adjust(right=0.99)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)

    # save the figure to html
    bio = BytesIO()
    plt.savefig(bio)
    plt.close()
    html = """<img src="data:image/png;base64,%s"/>""" % base64.b64encode(bio.getvalue()).decode('utf-8')
    return html


#######################
#  Pandas extension   #
#######################

@pd.api.extensions.register_dataframe_accessor("sf")
class PandasSpeechFirst:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def display(self, sparkline_columns=['data']):

        df = self._obj.copy()

        for col in sparkline_columns:
            df[col] = df[col].apply(make_sparkline)

        return display_dataframe(df)

    def debug(self, playable_wav_columns=['data'], sparkline_columns=['data'], ):

        df = pd.DataFrame()

        for col in playable_wav_columns:
            df[f'playable-{col}'] = self._obj[col].apply(make_playable_wav)

        for col in sparkline_columns:
            df[f'sparkline-{col}'] = self._obj[col].apply(make_sparkline)

        return display_dataframe(df)


#####################
#       Kedro       #
#####################

import yaml


def get_credentials(filepath, key):
    with open(filepath, 'r') as file:
        credentials = yaml.load(file, Loader=yaml.FullLoader)[key]
    return credentials


from pathlib import Path, PurePosixPath
from kedro.io import AbstractDataSet, PartitionedDataSet


class WavFile(AbstractDataSet):
    '''Used to load a .wav file'''

    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self) -> pd.DataFrame:
        df = pd.DataFrame({'file': [self._filepath],
                           'data': [load_wav(self._filepath)]})
        return df

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(str(self._filepath))

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)


class WavFileS3(AbstractDataSet):
    '''Used to load a .wav file from S3'''

    def __init__(self, filepath, credentials):
        self._filepath = PurePosixPath(filepath)
        self._s3 = s3fs.S3FileSystem(**credentials)

    def _load(self) -> pd.DataFrame:
        file = self._filepath.relative_to('s3:')

        df = pd.DataFrame({'file': [file],
                           'data': [load_wav(self._s3.open(file))]})
        return df

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(str(self._filepath))

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)


class WavFiles(PartitionedDataSet):
    '''Replaces the PartitionedDataSet.load() method to return a DataFrame.'''

    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)
        self._files = glob(filepath, recursive=True)

    def load(self) -> pd.DataFrame:
        '''Returns dataframe'''
        dict_of_data = super().load()

        df = pd.concat(
            [delayed() for delayed in dict_of_data.values()]
        )

        return df


class WavFilesS3(AbstractDataSet):
    '''Used to load a directory of .wav files'''

    def __init__(self, filepath, credentials):
        self._filepath = PurePosixPath(filepath)
        self._s3 = s3fs.S3FileSystem(**credentials)
        self._files = self._s3.glob(filepath)

    def _load(self) -> pd.DataFrame:
        df = pd.DataFrame({'file': [file for file in self._files],
                           'data': [load_wav(self._s3.open(file)) for file in self._files]})
        return df

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(str(self._filepath))

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(filepath=self._filepath)


########
#  S3  #
########

credentials = get_credentials(os.path.join(CONFIG_PATH, 'local/credentials.yml'), 'dev_s3')
