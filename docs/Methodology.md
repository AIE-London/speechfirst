# Data Science Report

Our Speech-First MVP leverages AI to provide an assessment for each utterance made by the app's users. This analysis and scores are intended to be used by the SLT to better understand:
• Which patients require SLT attention
• Longitudinal progress being made by service users
• The aspects of speech which the patients are struggling most struggling with


## Understanding User Needs – Service User App
For each service user utterance, the AI aims to estimate:

1. Duration of the utterance - Length of the audio recording
2. Duration of the utterance that contains speech - We use the recordings spectrogram
and a bespoke frequency-based voice activity detection to trim away the start and
end of the recording that does not appear to contain speech.
3. Duration of speech within the utterance - We use the pyin algorithm implemented
within the librosa python library to assess the recording in 23ms^[1] intervals (10). For each 23ms interval, we use the pyin estimation of the likelihood that the interval contains speech. We then add the durations of the intervals likely to contain speech.
4. If the utterance is a consonant sound - how clear* was this pronunciation
5. If the utterance is a word - how clear* was this word
6. What was the volume of speech across the recording
7. What was the pitch (f0) of speech across the recording
Note: An utterance is "clear" if it can be confidently & correctly identified as the speaker's intended utterance by a model trained to predict the intended utterance of self-identifying healthy speakers.


### Preprocessing
Prior to the audio data’s use for model training, we performed multiple pre-processing steps:
1. Conversion to WAV format (initial recordings are in MPEG)
2. Conversion from stereo to mono
3. Trimming audio using our bespoke audio trimming algorithm
4. Removal of low quality or non-phoneme sounds and label correction

### Accuracy of Sound Models
The accuracy of the consonant sound model i.e. the models ROC AUC (OVO), stands at 98.5%. The aim throughout the course of the project was to refresh and improve on a near-daily basis. A way to interpret the accuracy could be understood through the following example:
• If you pick two random consonant sound recordings from among the test data (a subset of the 2500+ recordings that were not used for model training)
• And you know that one of the two should have label X1 and the other label X2
• The model has a 98.5% chance to correctly assign each label

### Limitations & Source of Bias

#### Consonant Scope –
Our consonant clarity model is limited by the volume of data we've been able to gather. To date, we’ve obtained 100+ recordings for consonants: k, m, p, b, g, t, d, n. Thus limiting our ability to support additional consonants such as "sh", "z", "th", etc. The modelling process we've developed can be extended to support additional consonants as more data becomes available.

#### Consonant Pattern -
All consonant utterances collected have been of the form "Ah-{consonant}ah". This means that we are not able to support variations of this
e.g. "eh-peh", "gah", etc...
The modelling process we've developed can be extended to support additional patterns, but the data requirements would grow exponentially with additional target sounds.

### Speech Scoring
Our speech scoring (words and phrases) relies on the speech-to-text API provided by AWS (AWS Transcribe). This exposes our solution to the limitations of the AWS Transcribe service. Some independent studies have found the main speech-to-text Application Programming Interfaces (APIs) to provide inconsistent levels of accuracy.
