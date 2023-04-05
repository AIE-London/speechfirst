from typing import Dict, List
from functools import reduce
import numpy as np
import requests
import time
# import git

import s3fs
import boto3
import librosa

s3 = s3fs.S3FileSystem()

# stt := Speech-to-Text
stt_client = boto3.client('transcribe')  # GCP service is 30x faster

# tts := Text-to-Speech
tts_client = boto3.client('polly')

SAMPLE_RATE = 16_000  # Target Sample Rate


def load_wav(filepath: str, sr=SAMPLE_RATE, to_mono: bool = True) -> List[float]:
    assert isinstance(filepath, str), 'The `filepath` should be a string.'

    if filepath.startswith('s3://'):
        con = s3.open(filepath)
    else:
        con = open(filepath, 'rb')

    wav, sr = librosa.load(con, sr=sr)

    if to_mono:
        wav = librosa.to_mono(wav)

    return wav


def get_volume(wav: List[float], sr: int = SAMPLE_RATE) -> Dict[str, float]:
    '''Note: sr is ignored, just kept for API consistency'''

    amplitude = librosa.amplitude_to_db(wav, ref=0)
    ans = calc_stats(amplitude)
    return ans


def get_pitch(wav: List[float], sr: int = SAMPLE_RATE) -> Dict[str, float]:
    # See https://librosa.org/doc/0.8.0/generated/librosa.pyin.html?highlight=pyin#librosa.pyin

    kwargs = dict(
        y=wav,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=int(sr * 23 / 1000),  # 23ms - recommended for speech in documentation
        sr=sr
    )

    f0, voiced_bool, voiced_prob = librosa.pitch.pyin(**kwargs)

    # voice_freqs = [freq for freq, voiced in zip(f0, voiced_bool) if voiced]
    voice_freqs = [freq for freq, voiced in zip(f0, voiced_bool) if freq == freq]

    approx_freqs = [round(freq / 10, 0) * 10 for freq in voice_freqs]

    ans = calc_stats(approx_freqs)
    return ans


def get_duration(wav: List[float], sr: int = SAMPLE_RATE) -> float:
    '''Returns audio clip duration in seconds'''
    ans = librosa.get_duration(y=wav, sr=sr)
    return ans


#################
#    SCORING    #
#################

# In this section, `info` is expected to be
# the return payload with everything
# except the score itself.

def get_score(info: Dict) -> float:
    excercise_type = info['exerciseDetail']['type']

    if excercise_type == 'consonant':
        score = get_consonant_score(info)

    elif excercise_type == 'consonant-pair':
        score = get_consonant_pair_score(info)

    elif excercise_type == 'sustained-sound':
        score = get_sustained_sound_score(info)

    elif excercise_type == 'speech':
        score = get_speech_score(info)

    else:
        raise NotImplementedError()

    return score


def get_sustained_sound_score(info: Dict) -> float:
    volume = info['feedback_detail']['volume']
    score = min(volume['mean'] / 45, 1) ** 0.5
    return score


def get_consonant_score(info: Dict) -> float:
    target = info['exerciseDetail']['target']
    target_proba = info['feedback_detail']['phoneme'][target]
    return target_proba ** 0.5


def get_consonant_pair_score(info: Dict) -> float:
    target = info['exerciseDetail']['target']
    pair = get_consonant_pair(target)

    target_proba = info['feedback_detail']['phoneme'][target]
    pair_proba = info['feedback_detail']['phoneme'][pair]

    geometric_mean = np.sqrt(target_proba * (1 - pair_proba))  # Geometric mean
    return geometric_mean


def get_speech_score(info: Dict) -> float:
    job_name = info['feedback_detail']['transcription_job']
    full_transcript = get_full_transcript(job_name)
    transcript_score = get_full_transcript_score(full_transcript)
    return transcript_score


#################
#  Ease of use  #
#################

def get_consonant_pair(consonant: str):
    pairs = [
        ('k', 'g'),
        ('p', 'b'),
        ('t', 'd'),
        ('m', 'n')
    ]

    fast_lookup = {
        **{c1: c2 for c1, c2 in pairs},
        **{c2: c1 for c1, c2 in pairs},
    }

    return fast_lookup.get(consonant)


#################
#     STATS     #
#################

def calc_stats(signal):
    try:
        ans = {
            'mean': np.mean(signal),
            'median': np.median(signal),
            'q05': np.percentile(signal, 5),
            'q25': np.percentile(signal, 25),
            'q75': np.percentile(signal, 75),
            'q95': np.percentile(signal, 95),
            'count': len(signal)
        }
        
    except Exception:
        ans = {
            'mean': -1,
            'median': -1,
            'q05': -1,
            'q25': -1,
            'q75': -1,
            'q95': -1,
            'count': -1
        }

    ans = {k: float(round(v, 3)) for k, v in ans.items()}

    return ans


##################
#   Validation   #
##################

def validate_input(event):
    assert 'uid' in event.keys()
    assert 'wavFile' in event.keys()
    assert 'exerciseDetail' in event.keys()

    excercise_type = event['exerciseDetail']['type']
    assert excercise_type in [
        'consonant',
        'consonant-pair',
        'speech',
        'sustained-sound'], f'Excercise {excercise_type} is not supported.'


def validate_wav(wav: np.ndarray):
    assert isinstance(wav, np.ndarray), f'The wav file should be an np.ndarray, not {type(wav)}'
    assert wav.dtype == np.float32, f'The wav file should be np.float32, not {wav.dtype}'
    assert wav.ndim == 1, f'The wav file should a 1-dim np.array of floats. The array is wav.shape = {wav.shape} and wav.ndim = {wav.ndim}'
    assert wav.max() <= 1, f'It would be strange for a wav file to have a max value of {wav.max()}, please double check.'
    assert wav.min() >= -1, f'It would be strange for a wav file to have a min value of {wav.min()}, please double check.'
    assert -0.1 < wav.mean() < 0.1, f'It would be strange for a wav file to have a mean value of {wav.mean()}, please double check.'


##################
# Speech to Text #
##################

def start_transcription_job(audiofile: str, client=stt_client) -> str:
    job_name = str(hash(audiofile + str(np.random.rand())))

    client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': audiofile},
        MediaFormat='wav',
        LanguageCode='en-GB'
    )

    return job_name


def get_full_transcript(job_name, client=stt_client) -> Dict:
    job = client.get_transcription_job(TranscriptionJobName=job_name)
    uri = job['TranscriptionJob']['Transcript']['TranscriptFileUri']
    ans = requests.get(uri).json()
    return ans


def format_full_transcript(transcript):
    items = transcript['results']['items']

    words = [i for i in items if i['type'] == 'pronunciation']

    word_confidence = [float(w['alternatives'][0]['confidence']) for w in words]
    return word_confidence


def get_full_transcript_score(transcript):
    word_confidence = format_full_transcript(transcript)

    if len(word_confidence) > 0:
        score = reduce(lambda x, y: x * y, word_confidence) ** (1 / len(word_confidence))
    else: 
        score = 0 # No words detected
    return score


def get_transcript(job_name: str, client=stt_client, timeout=5):
    while timeout > 0:
        try:
            raw = get_full_transcript(job_name, client)
            break
        except:
            timeout -= 1
            time.sleep(1)

    try:
        best_guess = raw['results']['transcripts'][0]['transcript']
    except:
        best_guess = ''

    return best_guess


##################
#   CONSONANTS   #
##################

def get_consonant_prediction(wav):
    ...


##################
#    DEBUG       #
##################

def get_code_version():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def get_code_datetime():
    repo = git.Repo(search_parent_directories=True)
    return str(repo.head.commit.committed_datetime)


def get_clock_time():
    return time.time()
