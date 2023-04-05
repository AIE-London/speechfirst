"""This module contains everything needed for the phoneme/consonant model, such as:
- Data prep
- Model head definition
- Model wrappers
- Model helper functions
"""
from __future__ import division

import collections
from typing import List

import IPython
import functools
import math
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import tensorflow as tf
import tensorflow_hub as hub
import random
import scipy.io.wavfile
import tarfile
import time
import sys
from base64 import b64decode
import pandas as pd
from collections import OrderedDict

from dataset import DatasetStage, DATA_PATH

SEED = 42

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
assert tf.__version__.startswith('1.')
from tfdeterminism import patch

patch()


# @title Helper functions and classes
def normalized_read(filepath):
    """Reads and normalizes a wavfile."""
    try:
        _, data = scipy.io.wavfile.read(open(filepath, mode='rb'))
        samples_99_percentile = np.percentile(np.abs(data), 99.9)
    except:
        print(f'{filepath} could not be read, consider removing this file')
        return None
    normalized_samples = data / samples_99_percentile
    normalized_samples = np.clip(normalized_samples, -1, 1)
    return normalized_samples


class EmbeddingDataFileList(object):
    """Container that loads audio, stores it as embeddings and can
    rebalance it."""

    def __init__(self, filelist,
                 targets=None,
                 groups=None,
                 label_max=10000,
                 negative_label="negative",
                 silence_label="silence",
                 negative_multiplier=25,
                 target_samples=32000,
                 progress_bar=None,
                 embedding_model=None):
        """Creates an instance of `EmbeddingDataFileList`."""
        self._negative_label = negative_label
        self._silence_label = silence_label
        self._data_per_label = collections.defaultdict(list)
        self._groups_per_label = collections.defaultdict(list)
        self._filelist_per_label = filelist
        self._labelcounts = {}
        self._label_list = list(targets)
        self._group_list = set()  # auto-generated
        total_examples = sum([min(len(x), label_max) for x in filelist.values()])
        if negative_label in filelist.keys():
            total_examples -= min(len(filelist[negative_label]), label_max)
            total_examples += min(len(filelist[negative_label]), int(negative_multiplier * label_max))
        if silence_label in filelist.keys():
            total_examples -= min(len(filelist[silence_label]), label_max)
            total_examples += min(len(filelist[silence_label]), int(negative_multiplier * label_max))

        filelist = {k: v for k, v in filelist.items() if v != []}  # filter out empty labels

        print("loading %d examples" % total_examples)
        example_count = 0

        if not groups:
            groups = filelist

        for label in filelist:
            if label not in self._label_list:
                raise ValueError("Unknown label:", label)
            label_files = filelist[label]
            label_file_groups = groups[label]
            # random.shuffle(label_files)
            if label == negative_label or label == silence_label:
                multiplier = negative_multiplier
            else:
                multiplier = 1
            label_max_multiplied = int(label_max * multiplier)
            for wav_file, group in zip(label_files[:label_max_multiplied], label_file_groups[:label_max_multiplied]):
                # Read WAV file
                data = normalized_read(wav_file)

                # Padding
                required_padding = target_samples - data.shape[0]
                if required_padding > 0:
                    data = np.pad(data, (required_padding, required_padding), 'constant')

                # Update counts
                self._labelcounts[label] = self._labelcounts.get(label, 0) + 1

                # Embed
                if embedding_model:
                    data = embedding_model.create_embedding(data)[0][0, :, :, :]
                self._data_per_label[label].append(data)

                # Session ID for grouping (prevent splitting sessions across train/test)
                self._groups_per_label[label].append(group)
                self._group_list.add(group)

                # Progress bar
                if progress_bar is not None:
                    example_count += 1
                    progress_bar.update(progress(100 * example_count / total_examples))
        self._group_list = list(self._group_list)

    @property
    def labels(self):
        return self._label_list

    @property
    def groups(self):
        return self._group_list

    def get_label(self, idx):
        return self.labels.index(idx)

    def get_group(self, idx):
        return self.groups.index(idx)

    def _get_filtered_data(self, label, filter_fn, use_groups=True, with_filepath=False):
        """

        Returns: embedding, label (index), [group (idx)], [filepath]
        """
        if use_groups:
            label_idx = self.labels.index(label)
            if with_filepath:
                return [(filter_fn(data), label_idx, self.groups.index(group), filepath) for data, group, filepath in zip(
                    self._data_per_label[label], self._groups_per_label[label], self._filelist_per_label[label])]
            else:
                return [(filter_fn(data), label_idx, self.groups.index(group)) for data, group in zip(self._data_per_label[label], self._groups_per_label[label])]
        else:
            label_idx = self.labels.index(label)
            if with_filepath:
                return [(filter_fn(x), label_idx, filepath) for x, filepath in zip(self._data_per_label[label], self._filelist_per_label[label])]
            else:
                return [(filter_fn(x), label_idx) for x in self._data_per_label[label]]

    def _multiply_data(self, data, filelist, factor, groups=None):
        assert len(data) == len(filelist), (len(data), len(filelist))
        samples = int((factor - math.floor(factor)) * len(data))
        dfg = [data, filelist]
        if groups:
            assert len(filelist) == len(groups)
            dfg.append(groups)
        dfg_zip = list(zip(*dfg))
        mult_dfg_zip = int(factor) * dfg_zip + random.sample(dfg_zip, samples)
        mult_dfg = list(zip(*mult_dfg_zip))  # unzip
        return mult_dfg

    def multiclass_rebalance(self):
        # Get counts and max count
        counts = dict()
        for label in self.labels:
            counts[label] = self._labelcounts[label]
        max_count = max(counts.items(), key=lambda k: k[1])[1]
        # Resample
        for label in self.labels:
            # calc how many times to multiply
            factor = max_count / counts[label]
            if self.groups:
                self._data_per_label[label], self._filelist_per_label[label], self._groups_per_label[label] = self._multiply_data(
                    self._data_per_label[label], self._filelist_per_label[label], factor, groups=self._groups_per_label[label])
            else:
                self._data_per_label[label], self._filelist_per_label[label] = self._multiply_data(
                    self._data_per_label[label], self._filelist_per_label[label], factor)
            print(f'Rebalanced {label}: {self._labelcounts[label]:,} -> {len(self._data_per_label[label]):,} samples')
            self._labelcounts[label] = len(self._data_per_label[label])

    def full_rebalance(self, negatives, labeled):
        """Rebalances for a given ratio of labeled to negatives."""
        negative_count = self._labelcounts[self._negative_label]
        labeled_count = sum(self._labelcounts[key]
                            for key in self._labelcounts.keys()
                            if key not in [self._negative_label, self._silence_label])
        try:
            labeled_multiply = labeled * negative_count / (negatives * labeled_count)
        except ZeroDivisionError:
            raise ValueError(f'Got {negatives} negatives and {labeled_count} labeled count, so zero division error occurs.')
        for label in self._data_per_label:
            if label in [self._negative_label, self._silence_label]:
                continue
            if self.groups:
                self._data_per_label[label], self._filelist_per_label[label], self._groups_per_label[label] = self._multiply_data(
                    self._data_per_label[label], self._filelist_per_label[label], labeled_multiply, groups=self._groups_per_label[label])
            else:
                self._data_per_label[label], self._filelist_per_label[label] = self._multiply_data(
                    self._data_per_label[label], self._filelist_per_label[label], labeled_multiply)
            print(f'Rebalanced {label}: {self._labelcounts[label]:,} -> {len(self._data_per_label[label]):,} samples')
            self._labelcounts[label] = len(self._data_per_label[label])

    def get_all_data_shuffled(self, filter_fn, use_groups=True, with_filepath=False):
        """Returns a shuffled list containing all the data."""
        return self.get_all_data(filter_fn, shuffled=True, use_groups=use_groups, with_filepath=with_filepath)

    def get_all_data(self, filter_fn, shuffled=False, use_groups=True, with_filepath=False):
        """Returns a list containing all the data."""
        data = []
        for label in self._data_per_label:
            label_data = self._get_filtered_data(label, filter_fn, use_groups=use_groups, with_filepath=with_filepath)
            data += label_data
        if shuffled:
            random.shuffle(data)
        return data


def cut_middle_frame(embedding, num_frames, flatten=False):
    """Extrats the middle frames for an embedding."""
    left_context = (embedding.shape[0] - num_frames) // 2
    if flatten:
        return embedding[left_context:left_context + num_frames].flatten()
    else:
        return embedding[left_context:left_context + num_frames]


def get_filter_fn(context_size=16, flatten=False):
    return functools.partial(cut_middle_frame, num_frames=context_size, flatten=flatten)


def prep_data(df: pd.DataFrame, positive_labels: List[str] = None, speech_embedding_model=None,
              target_col: str = 'sound_name', group_col: str = 'group', filepath_col: str = 'local_filepath', use_groups: bool = True,
              label_max: int = 1000, negative_multiplier: float = 1, add_silence: bool = True,
              show_progress: bool = True) -> EmbeddingDataFileList:
    """

    Args:
        df: DataFrame with the data.
        speech_embedding_model: TFHub speech embedding model, if not specified will load it automatically.
        target_col: Column which is used as a target/label
        filepath_col: Column for the filepath.
        use_groups: If true, will use groups to avoid data leakage.
        label_max: Max number of samples for the positive labels.... (old) The number examples for each target word that should be loaded. A higher number for the training data will lead to a better model,
            but it will also take longer to load/train. A good starting point is 40.
            Small numbers for the eval data may result in easy / hard eval subsets that could give an incorrect impression of the model quality.
        negative_multiplier: How many more non target examples should be loaded.
            This is set to 25 by default as the speech commands dataset maps 25 words to negative. Also applies to silence examples.
        add_silence: If True, will use data/01_raw/silence directory for silent (negative) samples.
        show_progress: If True, will show a progress bar (IPython based).

    Returns:
        EmbeddingDataFileList.

    Examples:
        >>> use_groups = True
        >>> with_filepath = True
        >>> context_size = 16

        >>> all_data_edfl = prep_data(df, use_groups=use_groups, with_filepath=with_filepath)
        >>> filter_fn = get_filter_fn(context_size=context_size)
        >>> all_data = all_data_edfl.get_all_data_shuffled(filter_fn, use_groups=use_groups, with_filepath=with_filepath)
        >>> input_df = pd.DataFrame(all_data, columns=['X', 'y'] + (['groups'] if use_groups else []) + (['filepath'] if with_filepath else []))
        >>> labels = all_data_edfl.labels
    """
    if not speech_embedding_model:
        embedding_model_url = "https://tfhub.dev/google/speech_embedding/1"
        speech_embedding_model = TfHubWrapper(embedding_model_url)

    silence_dir = os.path.join(DATA_PATH, '01_raw', 'silence')
    if not positive_labels:
        positive_labels = df[target_col].unique()
    if add_silence and os.path.exists(silence_dir):
        all_sound_names = df[target_col].unique() + ["silence"]
    else:
        all_sound_names = df[target_col].unique()

    if show_progress:
        progress_bar = IPython.display.display(progress(0, 100), display_id=True)
    else:
        progress_bar = None

    all_example_files = collections.defaultdict(list)
    all_example_file_groups = collections.defaultdict(list)

    sound_name_count = 0
    for sound_name in all_sound_names:
        if sound_name in positive_labels:
            label = sound_name
        else:
            label = "negative"
        label_df  = df[df[target_col] == sound_name]
        all_example_files[label].extend(label_df[filepath_col])
        all_example_file_groups[label].extend(label_df[group_col])
        if show_progress and progress is not None:
            sound_name_count += 1
            progress_bar.update(progress(100 * sound_name_count / len(all_sound_names)))

    edfl = EmbeddingDataFileList(
        all_example_files, groups=all_example_file_groups, label_max=label_max,
        negative_multiplier=negative_multiplier,
        targets=all_example_files.keys(), embedding_model=speech_embedding_model,
        progress_bar=progress_bar)

    return edfl


def progress(value, maximum=100):
    return IPython.display.HTML("""
  <progress value='{value}' max='{max}' style='width: 80%'>{value}</progress>
    """.format(value=value, max=maximum))


def _fully_connected_model_fn(embeddings, num_labels):
    """Builds the head model and adds a fully connected output layer."""
    net = tf.compat.v1.layers.flatten(embeddings)
    logits = tf.compat.v1.layers.dense(net, num_labels, activation=None)
    return logits


framework = tf.contrib.framework
layers = tf.contrib.layers


def _conv_head_model_fn(embeddings, num_labels, context):
    """Builds the head model and adds a fully connected output layer."""
    activation_fn = tf.nn.elu
    normalizer_fn = functools.partial(
        layers.batch_norm, scale=True, is_training=True)
    with framework.arg_scope([layers.conv2d], biases_initializer=None,
                             activation_fn=None, stride=1, padding="SAME"):
        net = embeddings
        net = layers.conv2d(net, 96, [3, 1])
        net = normalizer_fn(net)
        net = activation_fn(net)
        net = layers.max_pool2d(net, [2, 1], stride=[2, 1], padding="VALID")
        context //= 2
        net = layers.conv2d(net, 96, [3, 1])
        net = normalizer_fn(net)
        net = activation_fn(net)
        net = layers.max_pool2d(net, [context, net.shape[2]], padding="VALID")
    net = tf.layers.flatten(net)
    logits = layers.fully_connected(
        net, num_labels, activation_fn=None)
    return logits


class HeadTrainer(object):
    """A tensorflow classifier to quickly train and test on embeddings.

    Only use this if you are training a very small model on a very limited amount
    of data. If you expect the training to take any more than 15 - 20 min then use
    something else.
    """

    def __init__(self, model_fn, input_shape, targets,
                 head_learning_rate=0.001, batch_size=64, metrics=['accuracy'], activation_fn='softmax'):
        """Creates a `HeadTrainer`.

        Args:
          model_fn: function that builds the tensorflow model, defines its loss
              and returns the tuple (predictions, loss, accuracy).
          input_shape: describes the shape of the models input feature.
              Does not include a the batch dimension.
          num_targets: Target number of keywords.
        """
        self._input_shape = input_shape
        self.targets = targets
        self._output_dim = len(targets)
        self._batch_size = batch_size
        self.metrics = metrics
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._feature = tf.compat.v1.placeholder(tf.float32, shape=([None] + input_shape))
            self._labels = tf.compat.v1.placeholder(tf.int64, shape=(None))
            module_spec = hub.create_module_spec(
                module_fn=self._get_headmodule_fn(model_fn, len(targets)))
            self._module = hub.Module(module_spec, trainable=True)
            logits = self._module(self._feature)
            if activation_fn == 'softmax':
                self._predictions = tf.nn.softmax(logits)
            elif activation_fn == 'sigmoid':
                self._predictions = tf.nn.sigmoid(logits)
            else:
                raise ValueError('No such activation fn')
            self._loss, self._metrics = self._get_loss(
                logits, self._labels, self._predictions, metrics=metrics)
            self._update_weights = tf.compat.v1.train.AdamOptimizer(
                learning_rate=head_learning_rate).minimize(self._loss)
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._sess.as_default():
            with self._graph.as_default():
                self._sess.run(tf.compat.v1.local_variables_initializer())
                self._sess.run(tf.compat.v1.global_variables_initializer())

    def _get_headmodule_fn(self, model_fn, num_targets):
        """Wraps the model_fn in a tf hub module."""

        def module_fn():
            embeddings = tf.compat.v1.placeholder(
                tf.float32, shape=([None] + self._input_shape))
            logit = model_fn(embeddings, num_targets)
            hub.add_signature(name='default', inputs=embeddings, outputs=logit)

        return module_fn

    def _get_loss(self, logits, labels, predictions, metrics=['accuracy']):
        """Defines the model's loss and accuracy."""
        xentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(input_tensor=xentropy_loss)
        metric_outs = []
        for metric in metrics:
            # TODO: gives precision/recall much higher than what it should be? many metrics are broken
            if metric == 'accuracy':
                metric_outs.append(tf.contrib.metrics.accuracy(tf.argmax(predictions, 1), labels))
            if metric == 'tp':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_true_positives(tf.argmax(predictions), labels))
            if metric == 'fp':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_false_positives(tf.argmax(predictions), labels))
            if metric == 'tn':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_true_negatives(tf.argmax(predictions), labels))
            if metric == 'fn':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_false_negatives(tf.argmax(predictions), labels))
            if metric == 'precision':  # TODO: Too high?
                metric_outs.append(tf.contrib.metrics.streaming_precision(tf.argmax(predictions, 1), labels))
            if metric == 'recall':  # TODO: Too high
                metric_outs.append(tf.contrib.metrics.streaming_recall(tf.argmax(predictions, 1), labels))
            if metric == 'auc':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_auc(predictions, tf.cast(tf.one_hot(labels, depth=1), tf.float32)))
            if metric == 'pearson':  # TODO: Broken
                metric_outs.append(tf.contrib.metrics.streaming_pearson_correlation(predictions, tf.cast(tf.one_hot(labels, depth=1), tf.float32)))
        return loss, metric_outs

    def save_head_model(self, save_directory):
        """Saves the model."""
        with self._graph.as_default():
            self._module.export(save_directory, self._sess)

        # Write targets
        targets_filepath = os.path.join(save_directory, 'targets.txt')
        with open(targets_filepath, 'w') as f:
            f.writelines(t + ('\n' if idx < (len(self.targets) - 1) else '') for idx, t in enumerate(self.targets))

    def _feature_transform(self, batch_features, batch_labels):
        """Transforms lists of features and labels into into model inputs."""
        return np.stack(batch_features), np.stack(batch_labels)

    def _batch_data(self, data, batch_size=None):
        """Splits the input data into batches."""
        batch_features = []
        batch_labels = []
        batch_size = batch_size or len(data)

        if isinstance(data, pd.DataFrame):
            data = data[['X', 'y']].to_numpy()

        for feature, label in data:
            if feature.shape != tuple(self._input_shape):
                raise ValueError(
                    "Feature shape ({}) doesn't match model shape ({})".format(
                        feature.shape, self._input_shape))
            if not 0 <= label < self._output_dim:
                raise ValueError('Label value ({}) outside of target range'.format(
                    label))
            batch_features.append(feature)
            batch_labels.append(label)
            if len(batch_features) == batch_size:
                yield self._feature_transform(batch_features, batch_labels)
                del batch_features[:]
                del batch_labels[:]
        if batch_features:
            yield self._feature_transform(batch_features, batch_labels)

    def epoch_train(self, data, epochs=1, batch_size=None):
        """Trains the model on the provided data.

        Args:
          data: List of tuples (feature, label) where feature is a np array of
              shape `self._input_shape` and label an int less than self._output_dim.
              can also be a dataframe
          epochs: Number of times this data should be trained on.
          batch_size: Number of feature, label pairs per batch. Overwrites
              `self._batch_size` when set.

        Returns:
          tuple of accuracy, loss;
              accuracy: Average training accuracy.
              loss: Loss of the final batch.
        """
        batch_size = batch_size or self._batch_size
        metric_lists = collections.defaultdict(list)
        for _ in range(epochs):
            for features, labels in self._batch_data(data, batch_size):
                run_out = self._sess.run(
                    [self._loss, *self._metrics, self._update_weights],
                    feed_dict={self._feature: features, self._labels: labels})
                loss = run_out[0]
                for out, metric in zip(run_out[1:-1], self.metrics):
                    metric_lists[metric].append(out)

        returns = [loss]
        for metric in self.metrics:
            returns.append(np.average(metric_lists[metric]))  # otherwise avg metric
        return tuple(returns)

    def test(self, data, batch_size=None):
        """Evaluates the model on the provided data.

        Args:
          data: List of tuples (feature, label) where feature is a np array of
              shape `self._input_shape` and label an int less than self._output_dim.
          batch_size: Number of feature, label pairs per batch. Overwrites
              `self._batch_size` when set.

        Returns:
          tuple of accuracy, loss;
              accuracy: Average training accuracy.
              loss: Loss of the final batch.
        """
        batch_size = batch_size or self._batch_size
        metric_lists = collections.defaultdict(list)
        for features, labels in self._batch_data(data, batch_size):
            run_out = self._sess.run(
                [self._loss, *self._metrics],
                feed_dict={self._feature: features, self._labels: labels})
            loss = run_out[0]
            for out, metric in zip(run_out[1:], self.metrics):
                metric_lists[metric].append(out)
        returns = [loss]
        for metric in self.metrics:
            returns.append(np.average(metric_lists[metric]))  # otherwise avg metric
        return tuple(returns)

    def infer_all(self, data, batch_size=None):
        """Run inference on the provided data.

        DO NOT USE, SUFFERS FROM PREDICTION DISCREPENCY BETWEEN DEPLOYMENT VERSION DUE TO BATCH NORM

        Args:
          data: List of tuples (feature, label) where feature is a np array of
              shape `self._input_shape` and label an int less than self._output_dim.
          batch_size: Number of feature, label pairs per batch. Overwrites
              `self._batch_size` when set.

        Returns:
            Predictions
        """
        batch_size = batch_size or self._batch_size
        outs = []
        for features, labels in self._batch_data(data, batch_size):
            run_out = self._sess.run(
                [self._predictions],
                feed_dict={self._feature: features})
            outs.append(run_out)

        return np.concatenate(outs, axis=1)[0]

    def infer(self, example_feature):
        """Runs inference on example_feature."""
        if example_feature.shape != tuple(self._input_shape):
            raise ValueError(
                "Feature shape ({}) doesn't match model shape ({})".format(
                    example_feature.shape, self._input_shape))
        return self._sess.run(
            self._predictions,
            feed_dict={self._feature: np.expand_dims(example_feature, axis=0)})


def plot_step(plot, max_data, data, train_results, test_results, cv_test_results=None, ax=None):
    if not ax:
        ax = plot.gca()
    #     plt.clf()
    ax.grid(True)
    plt.xlim(0, max_data)
    plt.ylim(0.0, 1.05)
    line_styles = ['-', '--', '-.', ':', '.', 'o']
    for metric, line in zip(train_results.keys(), line_styles):
        ax.plot(data, train_results[metric], f"bo{line}")
        ax.plot(data, train_results[metric], f"b{line}", label=f"train_{metric}")
        if test_results:
            ax.plot(data, test_results[metric], f"ro{line}")
            ax.plot(data, test_results[metric], f"r{line}", label=f"test_{metric}")
        if cv_test_results:
            ax.plot(data, cv_test_results[metric], f"go{line}", label=f"cv_test_{metric}")
            ax.plot(data, cv_test_results[metric], f"go{line}")

    plt.xlabel('number of examples trained on', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


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
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            embedding_module_spec = hub.load_module_spec(embedding_model_dir)
            embedding_module = hub.Module(embedding_module_spec)
            head_module_spec = hub.load_module_spec(head_model_dir)
            head_module = hub.Module(head_module_spec)
            self._samples = tf.placeholder(
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
                self._sess.run(tf.global_variables_initializer())

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
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            head_module_spec = hub.load_module_spec(head_model_dir)
            head_module = hub.Module(head_module_spec)
            #             self._samples = tf.placeholder(
            #                 tf.float32, shape=[1, None], name='audio_samples')
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
                self._sess.run(tf.global_variables_initializer())

    def infer(self, embedding):
        sess_output = self._sess.run(
            [self._predictions],
            feed_dict={self._embedding: embedding})
        return {target: float(value) for target, value in zip(self.targets, sess_output[0][0])}
