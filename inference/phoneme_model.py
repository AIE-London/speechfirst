from __future__ import division

import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import random

SEED = 42

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.compat.v1.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def dynamic_padding(inp, min_size, inp_rank, rank_idx_to_pad, pad_direction='both'):
    pad_size = min_size - tf.shape(inp)[rank_idx_to_pad]
    if pad_direction == 'both':
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
    elif pad_direction == 'left':
        pad_size_left = pad_size
        pad_size_right = 0
    elif pad_direction == 'right':
        pad_size_left = 0
        pad_size_right = pad_size
    else:
        raise ValueError('No such padding direction. Choose from "left", "right" or "both"')

    paddings = [[0, 0] for rank in range(inp_rank)]
    paddings[rank_idx_to_pad] = [pad_size_left, pad_size_right]
    return tf.pad(inp, paddings)


class FullModelWrapper(object):
    """A loads a save model classifier."""

    def __init__(self, embedding_model_dir, head_model_dir, context_size=16, activation_fn='softmax'):
        with open(os.path.join(head_model_dir, 'targets.txt')) as f:
            self.targets = [t.rstrip('\n') for t in f.readlines()]
        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            embedding_module_spec = hub.load_module_spec(embedding_model_dir)
            embedding_module = hub.Module(embedding_module_spec)
            head_module_spec = hub.load_module_spec(head_model_dir)
            head_module = hub.Module(head_module_spec)
            self._samples = tf.compat.v1.placeholder(
                tf.float32, shape=[1, None], name='audio_samples')

            # Samples need to be padded so that we can generate an embedding where 2nd rank has at least 'context_size' dimensions
            # The first embedding dimension requires 12400 samples within an audio file, every other dimension requires 1280 samples
            min_sample_size = 12400 + ((context_size - 1) * 1280)

            padded_samples = tf.cond(tf.less(tf.shape(self._samples)[1], min_sample_size),
                                     true_fn=lambda: dynamic_padding(self._samples, min_sample_size, inp_rank=2, rank_idx_to_pad=1),
                                     false_fn=lambda: self._samples)

            # Embed padded samples
            embedding = embedding_module(padded_samples)

            # We only use the middle 'context_size' dimensions in the 2nd rank of the embedding,
            # this way embeddings are of consistent length for every audio file
            slice_begin = (tf.shape(embedding)[1] - context_size) // 2
            sliced_embedding = tf.slice(embedding, [0, slice_begin, 0, 0], [-1, context_size, -1, -1])

            # Predict on the sliced embedding
            logits = head_module(sliced_embedding)
            if activation_fn == 'softmax':
                self._predictions = tf.nn.softmax(logits)
            elif activation_fn == 'sigmoid':
                self._predictions = tf.nn.sigmoid(logits)
            else:
                raise ValueError('No such activation fn')

            with self._sess.as_default():
                self._sess.run(tf.compat.v1.global_variables_initializer())

    def infer(self, samples):
        samples = samples.reshape((1, -1))
        sess_output = self._sess.run(
            [self._predictions],
            feed_dict={self._samples: samples})
        return {target: float(value) for target, value in zip(self.targets, sess_output[0][0])}


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
