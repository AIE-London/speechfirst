"""
This module contains everything that is needed for audio trimming.

TODO: get rid of Google VAD stuff since it is useless anyway
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import webrtcvad


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_segment_speech(data, aggressiveness=3, sample_rate=16_000, frame_duration_ms=30):
    frames = frame_generator(frame_duration_ms, data, sample_rate)
    frames = list(frames)
    vad = webrtcvad.Vad(aggressiveness)
    is_speeches = []
    for frame in frames:
        is_speech = int(vad.is_speech(frame.bytes, sample_rate))
        is_speeches.append(is_speech)
    return is_speeches


def trim_audio_vad(data, sample_rate=16_000, aggressiveness=3, verbose=0, min_duration=0.2):
    """

    Args:
        data: Audio data array (same as output of utils.load_wav(filepath)).
        sample_rate: Sample rate.
    Returns:
        Trimmed audio data
    """
    assert sample_rate == 16_000, 'Sample rate must be 16000 for this procedure to work'

    duration = librosa.get_duration(data, sample_rate)
    segmentation = vad_segment_speech(data, aggressiveness=aggressiveness, sample_rate=sample_rate)
    voice_block_indices = np.where(np.array(segmentation) == 1)
    try:
        voice_start_block_idx, voice_end_block_idx = voice_block_indices[0][0], voice_block_indices[0][-1]
    except IndexError:
        if verbose >= 1:
            print('No voice found')
        # if voice does never come in, we just return None for error handling down the pipeline
        return None

    voice_start_time = voice_start_block_idx * duration / (len(segmentation) - 1)
    voice_end_time = voice_end_block_idx * duration / (len(segmentation) - 1)
    voice_duration = voice_end_time - voice_start_time

    if voice_duration < min_duration:
        if verbose >= 1:
            print(
                f'Voice duration is too short {voice_start_time:.2f} - {voice_end_time:.2f} s (duration: {voice_duration:.2f}/{duration:.2f} s) ({(1 - (voice_duration / duration)) * 100:.2f}% reduction)')
        return None

    if verbose >= 1:
        print(f'Voice found within {voice_start_time:.2f} - {voice_end_time:.2f} s (duration: {voice_duration:.2f}/{duration:.2f} s) ({(1 - (voice_duration / duration)) * 100:.2f}% reduction)')
    return data[int(voice_start_time * sample_rate):int(voice_end_time * sample_rate)]


def trim_audio(data, sample_rate=16_000, voice_thresh=0.1, verbose=0, plot=False, min_duration=0.2, use_vad=True, vad_padding_duration=0.1, vad_aggressiveness=3):
    """

    Args:
        data: Audio data array (same as output of utils.load_wav(filepath)).
        sample_rate: Sample rate.
    Returns:
        Trimmed audio data
    """
    assert sample_rate == 16_000, 'Sample rate must be 16000 for this procedure to work'

    S_full, phase = librosa.magphase(librosa.stft(data))

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 0.1 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(0.1, sr=sample_rate)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_v = 10
    power = 2

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.

    dbs_per_freq = librosa.amplitude_to_db(S_foreground, ref=np.max)

    # 85 - 255 Hz are the fundamental frequencies of human voice (according to https://en.wikipedia.org/wiki/Voice_frequency)
    # So we're aiming to determine where's voice based on those frequencies
    freq_by_idx = dict(enumerate(librosa.fft_frequencies(sr=sample_rate, n_fft=dbs_per_freq.shape[0])))

    # Voice frequencies: 78.125 - 406.25 (Hz)
    start_voice_freq_idx = 5
    end_voice_freq_idx = 26
    if verbose >= 3:
        print(f'Voice frequencies: {freq_by_idx[start_voice_freq_idx]} - {freq_by_idx[end_voice_freq_idx]} (Hz)')

    duration = librosa.get_duration(data, sample_rate)

    dbs_per_voice_freq = dbs_per_freq[start_voice_freq_idx:end_voice_freq_idx, :]
    dbs_voice = dbs_per_voice_freq.sum(axis=0)
    dbs_voice_norm = dbs_voice - dbs_voice.min()
    dbs_voice_norm = dbs_voice_norm / dbs_voice_norm.max()
    dbs_voice_threshed = (dbs_voice_norm > voice_thresh).astype('uint8')

    # Use custom algorithm to find points at which voice comes in and comes out
    voice_block_indices = np.where(dbs_voice_threshed == 1)
    try:
        voice_start_block_idx, voice_end_block_idx = max(0, voice_block_indices[0][0] - 1), voice_block_indices[0][-1]
    except IndexError:
        # if voice does never come in, we just return None for error handling down the pipeline
        return None

    voice_start_time = voice_start_block_idx * duration / (len(dbs_voice_threshed))
    voice_end_time = voice_end_block_idx * duration / (len(dbs_voice_threshed))

    if use_vad:
        # Use Google's VAD to find points at which voice comes in and out
        # TODO: try passing S_foreground rather than data?
        vad_segmentation = vad_segment_speech(data, aggressiveness=vad_aggressiveness, sample_rate=sample_rate)

        vad_voice_block_indices = np.where(np.array(vad_segmentation) == 1)
        try:
            vad_voice_start_block_idx, vad_voice_end_block_idx = vad_voice_block_indices[0][0], vad_voice_block_indices[0][-1]
        except IndexError:
            print('No voice found (VAD)')
            # if voice does never come in, we just return None for error handling down the pipeline
            return None

        vad_voice_start_time = vad_voice_start_block_idx * duration / (len(vad_segmentation) - 1)
        vad_voice_end_time = vad_voice_end_block_idx * duration / (len(vad_segmentation) - 1)

        # Google's VAD needs some padding to work properly
        vad_voice_start_time_padded = vad_voice_start_time - vad_padding_duration
        vad_voice_end_time_padded = vad_voice_end_time + vad_padding_duration

        # Take an overlap of VAD and custom algorithm voice times
        voice_start_time = max(voice_start_time, vad_voice_start_time_padded, 0)
        voice_end_time = min(voice_end_time, vad_voice_end_time_padded, duration)

    voice_duration = voice_end_time - voice_start_time

    if min_duration and voice_duration < min_duration:
        if verbose >= 1:
            print(
                f'Voice duration is too short {voice_start_time:.2f} - {voice_end_time:.2f} s (duration: {voice_duration:.2f}/{duration:.2f} s) ({(1 - (voice_duration / duration)) * 100:.2f}% reduction)')
        return None

    if verbose >= 1:
        print(f'Voice found within {voice_start_time:.2f} - {voice_end_time:.2f} s (duration: {voice_duration:.2f}/{duration:.2f} s) ({(1 - (voice_duration / duration)) * 100:.2f}% reduction)')
    if plot:
        x = np.arange(len(dbs_voice_threshed)) * (duration / len(dbs_voice_threshed))
        if use_vad:
            x_vad = np.arange(len(vad_segmentation)) * (duration / len(vad_segmentation))
        dbs = librosa.amplitude_to_db(S_foreground, ref=np.max).sum(axis=0)
        dbs_norm = dbs - dbs.min()
        dbs_norm = dbs_norm / dbs_norm.max()

        plt.figure(figsize=(20, 10))
        plt.plot(x, dbs_norm, label='foreground frequency amplitude (norm)', alpha=.4)
        plt.plot(x, dbs_voice_norm, label='voice frequency amplitude (norm)')
        if use_vad:
            plt.step(x_vad, vad_segmentation, label='VAD segmentation', linestyle='-.', color='red')
        plt.hlines(y=voice_thresh, xmin=voice_start_time, xmax=voice_end_time, colors='green', linestyles='-', lw=2, label='voice threshold')
        plt.vlines(x=voice_start_time, ymin=0, ymax=1, colors='black', linestyles='-', lw=2, alpha=1, label='voice start')
        plt.vlines(x=voice_end_time, ymin=0, ymax=1, colors='black', linestyles='-', lw=2, alpha=1, label='voice end')
        plt.step(x, dbs_voice_threshed, linestyle='--', color='green', label='above voice threshold', alpha=0.5)
        plt.legend()
    return data[int(voice_start_time * sample_rate):int(voice_end_time * sample_rate)]
