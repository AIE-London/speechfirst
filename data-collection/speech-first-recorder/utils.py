import base64
import os
import re
import wave
import contextlib
import streamlit as st


def get_valid_filename(s):
    """Changes string so it becomes a valid filename by replacing invalid characters

    Args:
        s: String

    Returns:
        Valid filename
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def substitute_template_variables(code, mappings):
    """Replace template variable such as {{ var }} with a value within JS code

    Args:
        code: JavaScript code template
        mappings: Dictionary mapping template variables to values

    Returns:
        JavaScript code where variable is replaced with a value
    """
    for var, val in mappings.items():
        code = re.sub('{{' + fr'\s*{var}\s*' + '}}', str(val), code, flags=re.IGNORECASE)
    return code


def get_audio_html(audio_filepath):
    """Get HTML code that plays an audio

    Args:
        audio_filepath: Path to audio that will be played.

    Returns:
        HTML
    """
    fmt = os.path.splitext(audio_filepath)[-1][1:]
    with open(audio_filepath, 'rb') as f:
        audio_bytes = f.read()
    return f"""
    <audio autoplay class="stAudio">
    <source src="data:audio/{fmt};base64,{base64.b64encode(audio_bytes).decode()}" type="audio/{fmt}">
    Your browser does not support the audio element.
    </audio>
"""


def get_download_link(filename) -> str:
    obj = open(filename, 'rb').read()
    b64 = base64.b64encode(obj).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Zip File</a>'
    return href


def get_audio_duration(filepath):
    """Get duration of the audio file (WAV)

    Args:
        filepath: Path to audio file

    Returns:
        Duration in seconds
    """
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        seconds = frames / float(rate)
        return seconds
