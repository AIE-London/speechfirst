import numpy as np
import streamlit as st
import time
import os
from glob import glob
import shutil
from datetime import datetime
import streamlit.components.v1 as components
import yaml
from utils import substitute_template_variables, get_valid_filename, get_audio_html, get_download_link, get_audio_duration
from streamlit.report_thread import get_report_ctx

# Current context. Useful for grabbing a unique ID
ctx_id = get_report_ctx().session_id

with open('config.yml') as f:
    config = yaml.load(f, yaml.SafeLoader)

with open('record.js', 'r') as f:
    js_code_template = f.read()

audio_notif_start_html = get_audio_html('click1.wav')
audio_notif_end_html = get_audio_html('click2.wav')
error_notif = st.empty()


def play_reference_sound(dataset, sound_name, delay_after=float(config.get('delay_after_reference_sound')), add_audio_time_to_delay=True):
    """Play the reference sound (sound that user will have to say)

    Args:
        dataset: Dataset name
        sound_name: Sound name
        delay_after: How many seconds to wait for after the start of the sound.
            If None, will use the sound length.
    """
    path = os.path.join('reference_sounds', dataset, get_valid_filename(sound_name) + '.wav')
    if os.path.exists(path):
        ref_sound_html = get_audio_html(path)
        components.html(ref_sound_html, width=0, height=0)

        if add_audio_time_to_delay is True:
            delay_after += get_audio_duration(path)
        time.sleep(delay_after)


def record(name, dataset, session_id, wait_time, play_sound: bool = True, save_enabled: bool = True):
    """Record a single sound

    Args:
        name: Name of the recording
        dataset: Dataset name
        session_id: Session ID
        wait_time: Number of seconds to wait for the recording
    """
    mappings = dict(NAME=name,
                    DATASET=dataset,
                    SESSION_ID=session_id,
                    SERVER_URL=config.get('api_url'),
                    PLAYBACK_ENABLED=int(bool(config.get('playback_enabled'))),
                    SAVE_ENABLED=int(bool(save_enabled)),
                    MILLISECONDS=wait_time * 1000)

    js_code = substitute_template_variables(js_code_template, mappings)
    components.html(f'''<script>{js_code}</script>''', width=0, height=0)
    if play_sound is True:
        components.html(audio_notif_start_html, width=0, height=0)
    time.sleep(wait_time)
    if play_sound is True:
        components.html(audio_notif_end_html, width=0, height=0)


st.title('Help us gather audio data!')

st.write('''Please help us gather some data by recording yourself saying some sounds. This way we can build an AI model that will help patients practice speech therapy.''')
st.write('''The recordings will only be used to teach AI what is a valid pronunciation and won't be played back to anyone.''')

session_id = ctx_id
session_id = get_valid_filename(session_id)  # Make sure we can convert this does not contain anything that would break things

if bool(config.get('allow_dataset_choice') is True):
    dataset_choices = [os.path.splitext(os.path.basename(sn_filepath))[0] for sn_filepath in os.listdir('sound_names') if os.path.splitext(sn_filepath)[-1] == '.txt']

    try:
        default_dataset_index = [idx for idx, dataset_choice in enumerate(dataset_choices) if dataset_choice == config.get('default_dataset')][0]
    except:
        default_dataset_index = 0

    dataset = st.selectbox('Select a dataset type that you would like to record', options=dataset_choices, index=default_dataset_index)
else:
    dataset = config.get('default_dataset')

with open(os.path.join('sound_names', dataset + '.txt'), 'r') as f:
    sound_names = [name.strip() for name in f.readlines()]

sample_count = st.number_input(f'There are {len(sound_names)} sounds to be recorded. How many samples are you willing to record for each sound?', value=int(config.get('default_sample_count')), step=1,
                               min_value=1)
sample_time = st.number_input('How many seconds do you need for each sample?', value=float(config.get('default_sample_time')), min_value=1., max_value=5., step=0.1)
sample_count_digit_count = 1 + int(np.log10(sample_count))

session_path = os.path.join(config.get('recordings_dir'), dataset, session_id)

if os.path.exists(session_path):
    error_notif.markdown('<span style="color:red">Please choose a different session ID. This one already exists in our system.</span>', unsafe_allow_html=True)

st.markdown('### Instructions')
st.write(f'When you press record, you will be asked to say each sound {sample_count} times. '
         f'''Each recording starts with a click sound, so make sure that you only talk after it. If you made a mistake, you can re-start the session by pressing 'Start' button again.''')
st.write(f'''**Make sure you only use Chrome browser for the application to work correctly**. When you press 'Start' your browser will ask for microphone permissions. 
Tick "Remember this decision" box if it is available (to avoid getting repeated microphone permission prompts) and allow the permissions. Then, press 'Start' to re-start the session.''')
st.write(f'''Recordings will automatically get saved on our end, so there is no need to send it to us.''')

button_cols = st.beta_columns(5)

with button_cols[0]:
    start = st.button('Start')
with button_cols[1]:
    stop = st.button('Stop')
with button_cols[2]:
    review = st.button('Review')
with button_cols[3]:
    download = st.button('Download')
with button_cols[4]:
    delete = st.button('Delete')

dynamic_el_1 = st.empty()
dynamic_el_2 = st.empty()

if start:
    dynamic_el_1.markdown(f'#### <span style="color:red">Please enable microphone permissions (make sure to tick "Remember this decision" option if available)</span>', unsafe_allow_html=True)
    dynamic_el_2.markdown(f'Speech recording is about to start...', unsafe_allow_html=True)
    record('init', dataset=dataset, session_id=session_id, wait_time=5, play_sound=False, save_enabled=False)
    for sound_idx, sound_name in enumerate(sound_names):
        dynamic_el_1.markdown(f'# {sound_name} ({sound_idx + 1}/{len(sound_names)})')
        play_reference_sound(dataset, sound_name)
        for sample_idx in range(sample_count):
            time.sleep(config.get('start_delay'))
            dynamic_el_1.markdown(f'# <span style="color:green">{sound_name}</span> ({sound_idx + 1}/{len(sound_names)}) ({sample_idx + 1}/{sample_count})', unsafe_allow_html=True)
            dynamic_el_2.markdown(f'### Say **"{sound_name}"**')
            name = f'{get_valid_filename(sound_name)}-{str(sample_idx + 1).rjust(sample_count_digit_count, "0")}'
            record(name, dataset=dataset, session_id=session_id, wait_time=sample_time)
            dynamic_el_1.markdown(f'# <span style="color:red">{sound_name}</span> ({sound_idx + 1}/{len(sound_names)}) ({sample_idx + 1}/{sample_count})', unsafe_allow_html=True)
            dynamic_el_2.markdown(f'# 🤫🤐😶')
            time.sleep(config.get('stop_delay'))

    dynamic_el_1.markdown(f'### Session has been completed ✅')
    dynamic_el_2.markdown('''The recordings have been successfully saved on our end. 
    You can review/download/delete them using the buttons above. Thank you for your time and have a great day!''')

if review:
    files = glob(os.path.join(session_path, 'wav/') + '*.wav')
    print(files)
    cols = st.beta_columns(sample_count)

    for sample_idx, file in enumerate(files):
        with cols[sample_idx % sample_count] as c:
            st.text(os.path.basename(file))
            st.audio(file, format='audio/wav', start_time=0)

if delete:
    os.system(f'rm -rf {session_path}')
    dynamic_el_1.markdown('### Your data has been deleted')
    dynamic_el_2.markdown('')

if download:
    dynamic_el_1.markdown('### Preparing your downloading link')
    zip_basename = f'{dataset}-{session_id}'
    shutil.make_archive(zip_basename, 'zip', session_path)
    file = sorted(glob(f'{dataset}-{session_id}.zip'))[-1]
    href = get_download_link(file)
    dynamic_el_1.markdown('Your file is ready!')
    dynamic_el_2.markdown(href, unsafe_allow_html=True)
