from typing import List

import pyaudio
import wave
import time
import os

from audiofile import AudioFile


class Recorder:
    def __init__(self, start_notification_sound_path: str = 'click1.wav',
                 end_notification_sound_path: str = 'click2.wav',
                 channels: int = 1,
                 sample_rate=16_000):
        """Class for records sounds

        Args:
            start_notification_sound_path: Sound to play at the start of each recording
            start_notification_sound_path: Sound to play at the end of each recording
            channels: 1 for mono, 2 for stereo
            sample_rate: Sample rate (16000 = 44100 samples per second)
        """

        self.start_notification_sound_path = start_notification_sound_path
        self.end_notification_sound_path = end_notification_sound_path
        self.channels = channels
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()  # Create an interface to PortAudio

    def record(self, name, path, seconds) -> None:
        """Record a sound to a file

        Args:
            path: Path to the recording (will be created)
            seconds: Number of seconds for a file
        """
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16

        # Notify user of recording start
        print(f'Recording "{path}" for {seconds} seconds')
        print(f'Say {name}')
        if self.end_notification_sound_path:
            AudioFile(self.start_notification_sound_path).play()
        time.sleep(0.15)  # to make sure the start sound is not being recorded

        stream = self.p.open(format=sample_format,
                             channels=self.channels,
                             rate=self.sample_rate,
                             frames_per_buffer=chunk,
                             input=True)

        # Store data in chunks
        frames = []
        for i in range(0, int(self.sample_rate / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Notify user of recording end
        print('Finished recording')
        if self.end_notification_sound_path:
            AudioFile(self.end_notification_sound_path).play()

        # Save the recorded data as a WAV file
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(sample_format))
        wf.setframerate(self.sample_rate)
        audio_bytes = b''.join(frames)
        wf.writeframes(audio_bytes)
        wf.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Terminate the PortAudio interface
        self.p.terminate()

    def capture_data(self, directory: str, sound_names: List[str], sample_duration: int, sample_count: int) -> None:
        """Capture data for each sound name

        Args:
            directory: Directory where to save the files.
            sound_names: Names of the sounds to capture.
            sample_duration: Duration of each sample.
            sample_count: Number of samples to capture for each sound.
        """
        os.makedirs(directory, exist_ok=True)

        def count_in():
            for i in ['.']:
                print(i, end=' ')
                time.sleep(0.5)
            print()

        for sound_name in sound_names:
            print('#######################')
            print(f'Say:  {sound_name} ')
            print('#######################')

            for i in range(sample_count):
                count_in()
                filename = f'''{directory}/{sound_name}-{str(i).rjust(4, str(0))}.wav'''
                self.record(sound_name, filename, sample_duration)

            print("===END===")
            time.sleep(2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='directory where to record the files to')
    parser.add_argument('--sample_count', help='number of samples per sound', default=10)
    parser.add_argument('--duration', help='duration of each samples in seconds', default=1)
    parser.add_argument('--channels', help='number of channels')
    parser.add_argument('--sound_names', help='filepath to a text file with sound names to record (separated by lines)', default='sound_names/alphabet.txt')
    args = parser.parse_args()

    r = Recorder()

    with open(args.sound_names, 'r') as f:
        sound_names = [name.strip() for name in f.readlines()]

    r.capture_data(args.directory, sound_names=sound_names, sample_count=args.sample_count, sample_duration=args.duration)
