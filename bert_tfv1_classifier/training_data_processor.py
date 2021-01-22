
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import logging
import collections

import csv
import pandas as pd

import tensorflow as tf

from models import tokenization
from models.model_classifier import convert_single_example_tokens

#
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
#
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
#
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    #
#
class SimProcessor(DataProcessor):
    """Processor for the Sim task"""
    #
    def get_train_examples(self, data_dir):
        """
        """
        data_tag = "train"
        file_path = os.path.join(data_dir, 'train_sentiment.txt')
        data_loaded = []
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        index = 0
        for line in lines:
            guid = '%s-%d' % (data_tag, index)
            line = line.replace("\n", "").split("\t")
            #
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            #
            data_loaded.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
            #
        #
        return data_loaded
        #
    #
    def get_dev_examples(self, data_dir):
        """
        """
        data_tag = "valid"
        file_path = os.path.join(data_dir, 'test_sentiment.txt')
        data_loaded = []
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        index = 0
        for line in lines:
            guid = '%s-%d' % (data_tag, index)
            line = line.replace("\n", "").split("\t")
            #
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            #
            data_loaded.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
            #
        #
        return data_loaded
        #
    #
    def get_test_examples(self, data_dir):
        """
        """
        data_tag = "test"
        file_path = os.path.join(data_dir, 'test_sentiment.txt')
        data_loaded = []
        #
        fp = open(file_path, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        #
        index = 0
        for line in lines:
            guid = '%s-%d' % (data_tag, index)
            line = line.replace("\n", "").split("\t")
            #
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            #
            data_loaded.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
            #
        #
        return data_loaded
        #
    #
    def get_labels(self):
        return ['0', '1', '2']
    #
    def get_class_weights(self):
        return [1, 1, 3]
    #
#

#
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, logger=None):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    #
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    #
    label_id = label_map[example.label]
    #
    input_ids, input_mask, segment_ids, label_id = convert_single_example_tokens(
        tokenizer, tokens_a, tokens_b, label_id, max_seq_length
    )
    #

    #
    if ex_index < 5 and logger:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_a]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    #
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    #
    return feature
    #
#
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, logger=None):
    """Convert a set of `InputExample`s to a TFRecord file."""
    #
    writer = tf.python_io.TFRecordWriter(output_file)
    #
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and logger:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, logger)
        #
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        #
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        #
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        #
    #
    writer.close()
    #
#
def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """ return: function dataset_creator(settings)
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    #
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]
        batch_size = params.batch_size

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d
    #
    return input_fn
    #
#

