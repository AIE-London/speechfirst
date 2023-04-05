# SpeechFirst

SpeechFirst was a "Tech for Positive Futures" initiative from a team at Capgemini. Our goal was to apply machine learning and artificial intelligence to address the shortage of speech and language therapists in the UK.

During the course of the project we sought to prove the viability of evaluating speech with machine learning. Now we want to open-source our learnings, such that others can build upon the work we did to further research this field.

### Scope and Limitations

We exclusively built tooling to evaluate:

- Consonant sound replication accuracy
- Phoneme identification and accuracy
- Pitch consistency
- Volume consistency
- Transcription comparison

In addition to these individual metrics, we devised a mechanism for computing a score that combined any of the above metrics, where appropriate for a given task or exercise.

We are not publishing any pre-trained models. This repository serves as our research, not a ready-for-usage product and is released under MIT licensing without warranty or liability.

### Content

We're providing our modelling resources, data collection tools and inference code. We won't be providing any pre-trained models or data.

Within this project, there are three subdirectories:
- data-collection
- modeling
- inference

Read the readme.md in each directory for a guide on how to get the relevant step running.

You will need to begin with data collection, bring that data into modelling and then finally leverage the trained models you've produced in inference.
