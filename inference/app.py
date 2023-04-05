from trim import trim_audio
from utils import *
from phonemes import get_phoneme_prediction
import json
from consonant_sound_detector import get_probability_of_consonant_sound_isf, get_probability_of_consonant_sound_clf, get_speech_embedding_model, embed_audio, cut_middle_frame, normalize_audio, \
    pad_audio

CONTEXT_SIZE = 16

def handler(event, context):

    start_time = get_clock_time()

    file = event['wavFile'] # This should be a filepath to S3
    wav = load_wav(file) # Load from S3 as a numpy array (default sample rate is 16k)
    wav_trimmed = trim_audio(wav)
    
    volume = get_volume(wav)
    duration = get_duration(wav)
    speech_duration = get_duration(wav_trimmed)
    pitch = get_pitch(wav)

    wav_n = normalize_audio(wav_trimmed)
    wav_p = pad_audio(wav_n)

    embedding_model = get_speech_embedding_model()
    wav_e = embed_audio(wav_p, embedding_model)

    wav_m = cut_middle_frame(wav_e, CONTEXT_SIZE, flatten=False)

    phoneme = get_phoneme_prediction(wav_m)

    p_consonat_isf = get_probability_of_consonant_sound_isf(wav_m)
    p_consonat_clf = get_probability_of_consonant_sound_clf(wav_m)

    if event['exerciseDetail']['type'] == 'speech':
        transcription_job = start_transcription_job(file)
        transcript = get_transcript(transcription_job, timeout=120)
    else:
        transcription_job = None
        transcript = None

    analysis = {
        "feedback_detail": {
            "duration": duration,
            "speech_duration": speech_duration,
            "volume": volume,
            "pitch": pitch,
            "transcript": transcript,
            "transcription_job": transcription_job,
            "phoneme": phoneme
          }
        }

    dictionary_response = {**event, **analysis}

    dictionary_response['feedback'] = {
        "score": get_score(dictionary_response)
    }

    # This code is for checking if the sound is a consonant of some type, and scoring based on that
    #if event['exerciseDetail']['type'] in ['consonant', 'consonant-pair']:
    #    x = dictionary_response['feedback']['score']
    #    dictionary_response['feedback']['score'] = ((x* (p_consonat_isf*p_consonat_clf)**.5)**.5)**.25

    end_time = get_clock_time()

    dictionary_response['debug'] = {
        'runtime': end_time - start_time,
        'ifs_consonant_prob': p_consonat_isf,
        'cls_consonant_prob': p_consonat_clf
    }

    #response = json.dumps(dictionary_response, indent = 4) 
    response = dictionary_response  
    return response