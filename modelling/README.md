# SpeechFirst Modelling

This area of our repository contains the tools necessary to train an instance of a phoneme identification model.

### Dependencies

#### Dataset

In order to train the models, you will need an appropriate dataset, as collected via the "data-collection" subdirectory of this project.

#### Model Weights

In order to process the data and train the phoneme model, we require two pretrained model's weights to be downloaded.

##### Speech Embedding

Download the Speech Embedding model from TFHub (https://tfhub.dev/google/speech_embedding/1)

The "saved_model.pb" should be placed under the `modelling/phoneme-model/speech_embedding_1` directory and the variables files in a subdirectory of this named "variables".

### Training Models

#### Consonants Classifier

Begin by training the consonants classifier using the python notebook in the "consonants-model" directory

Once complete, you should save the weights and ensure you place them under the `modelling/phoneme-model/anomaly_voice_convolutional_softmax` directory and the variables files in a subdirectory of this named "variables".

#### Phoneme Model

Given a trained consonants classifier placed under the `modelling/phoneme-model/anomaly_voice_convolutional_softmax` directory

And given the Speech Embedding model placed under `modelling/phoneme-model/speech_embedding_1`.

You can access the following series of notebooks to train the phoneme model.

1. "embed_wav.ipynb" - Generates embeddings from your training data
2. "phoneme_model.ipynb" - Trains a model to identify phonemes from the training data and embeddings you've created.
3. "identify_mislabelled_data.ipynb" - Can help with improving data quality by identifying mislabled data.

Once both of these models are ready, place them under `inference/phoneme_convolutional_softmax` and `inference/anomaly_voice_convolutional_softmax`.

You should be ready for inference.

