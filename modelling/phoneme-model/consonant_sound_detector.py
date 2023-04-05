import tensorflow as tf
import tensorflow_hub as hub
import os
import scipy
import numpy as np


class TfHubWrapper(object):
    """A loads a tf hub embedding model."""

    def __init__(self, embedding_model_dir):
        """Creates a `SavedModelWraper`."""
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            with self._sess.as_default():
                module_spec = hub.load_module_spec(embedding_model_dir)
                embedding_module = hub.Module(module_spec)
                self._samples = tf.compat.v1.placeholder(
                    tf.float32, shape=[1, None], name='audio_samples')
                self._embedding = embedding_module(self._samples)
                self._sess.run(tf.compat.v1.global_variables_initializer())
        print("Embedding model loaded, embedding shape:", self._embedding.shape)

    def create_embedding(self, samples):
        samples = samples.reshape((1, -1))
        output = self._sess.run(
            [self._embedding],
            feed_dict={self._samples: samples})
        return output


def normalize_audio(wav):
    """Normalizes a wavfile.
    
    It makes it so that most 99.9% of the data is 
    between -1 and 1, and the .1% is cliped to 1,-1 as appropiate."""
    samples_99_percentile = np.percentile(np.abs(wav), 99.9)
    normalized_samples = wav / samples_99_percentile
    normalized_samples = np.clip(normalized_samples, -1, 1)
    return normalized_samples


def get_speech_embedding_model():
    embedding_model_url = 'speech_embedding_1'
    return TfHubWrapper(embedding_model_url)


def pad_audio(wav):
    target_samples = 32000
    required_padding = target_samples - wav.shape[0]
    if required_padding > 0:
        padded_data = np.pad(wav, (required_padding, required_padding), 'constant')
    else:
        padded_data = wav
    return padded_data


def embed_audio(wav, speech_embedding_model):
    emb = speech_embedding_model.create_embedding(wav)[0][0, :, :, :]
    return emb


def cut_middle_frame(embedding, num_frames, flatten=False):
    """Extrats the middle frames for an embedding."""

    left_context = (embedding.shape[0] - num_frames) // 2
    if flatten:
        return embedding[left_context:left_context + num_frames].flatten()
    else:
        return embedding[left_context:left_context + num_frames]


def pred_is_phoneme(emb_middle_flat):
    import pickle
    pkl_filepath = "phoneme_or_not_isolation_forest.pkl"
    with open(pkl_filepath, 'rb') as file:
        isf = pickle.load(file)

    isf_preds = isf.decision_function([emb_middle_flat])  # it is also possible to feed a batch of embeddings
    return isf_preds


def get_probability_of_consonant_sound_isf(wav_m):
    p = pred_is_phoneme(wav_m.ravel())[0]

    score = 0.5 * np.tanh(3 / 0.15 * p) + .5
    return score


class ClassificationModelWrapper(object):
    """Like FullModelWrapper but takes in sliced embedding directly"""

    def __init__(self, head_model_dir, context_size=16, activation_fn='softmax'):
        with open(os.path.join(head_model_dir, 'targets.txt')) as f:
            self.targets = [t.rstrip('\n') for t in f.readlines()]
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            head_module_spec = hub.load_module_spec(head_model_dir)
            head_module = hub.Module(head_module_spec)

            self._embedding = tf.compat.v1.placeholder(
                tf.float32, shape=[None, context_size, 1, 96]
            )

            logits = head_module(self._embedding)
            if activation_fn == 'softmax':
                self._predictions = tf.nn.softmax(logits)
            elif activation_fn == 'sigmoid':
                self._predictions = tf.nn.sigmoid(logits)
            else:
                raise ValueError('No such activation fn')

            with self._sess.as_default():
                self._sess.run(tf.compat.v1.global_variables_initializer())

    def infer(self, embedding):
        sess_output = self._sess.run(
            [self._predictions],
            feed_dict={self._embedding: embedding})
        return {target: float(value) for target, value in zip(self.targets, sess_output[0][0])}


def get_probability_of_consonant_sound_clf(wav_m):
    clf = ClassificationModelWrapper('anomaly_voice_convolutional_softmax')
    clf_preds = clf.infer([wav_m])
    p = clf_preds['negative']
    return p
