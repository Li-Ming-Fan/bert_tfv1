#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:19:42 2019
@author: li-ming-fan
"""

import os
import numpy as np

import re
import collections

import time
import logging
import json

import tensorflow as tf
from tensorflow.python.framework import graph_util

try:
    from modeling_bert import BertConfig, BertModel
except:
    from .modeling_bert import BertConfig, BertModel

#
def model_forward_procedure(bert_config, is_training, 
                 input_ids, input_mask, segment_ids, labels, 
                 upper_settings):
    """ Creates a classification model. """
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)
    #
    output_layer = model.get_pooled_output()
    #

    #
    num_labels = upper_settings["num_labels"]
    class_weights = upper_settings["class_weights"]
    #

    #
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        if is_training:
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            if class_weights:
                tensor_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
                tensor_weights = tf.expand_dims(tensor_weights, 0)   # [1, C] for [B, C]
                one_hot_labels = one_hot_labels * tensor_weights
            else:
                pass

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            #
            out_dict = {}
            out_dict["loss_train"] = loss
            out_dict["per_example_loss"] = per_example_loss
            out_dict["logits"] = logits
            out_dict["probabilities"] = probabilities
            #
            return out_dict
            #
        else:
            out_dict = {}
            out_dict["probabilities"] = probabilities
            #
            return out_dict
            #
        #
    #
#
def model_graph_creator(settings, input_tensors=None):
    """
    """
    if input_tensors is None:
        input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')
        #
        is_training = False
        label_ids = None
        #
    else:
        is_training = True
        #
        input_ids = input_tensors["input_ids"]
        input_mask = input_tensors["input_mask"]
        segment_ids = input_tensors["segment_ids"]
        label_ids = input_tensors["label_ids"]
        #
    #

    #
    bert_config = settings.bert_config
    upper_settings = settings.upper_settings
    #
    outputs = model_forward_procedure(bert_config, is_training, 
                           input_ids, input_mask, segment_ids, label_ids, 
                           upper_settings)
    #
    print("model defined, with inputs and outputs:")
    #
    inputs = input_ids, input_mask, segment_ids
    for item in inputs:
        print(item)
    if isinstance(outputs, tf.Tensor): 
        print(outputs)
    else:
        for item in outputs:
            print(item)
    #
    return outputs
    #
#

#
def convert_single_example_tokens(tokenizer, tokens_a, tokens_b=None, label_id=None,
                        max_seq_length=128, logger=None):
    """
    """
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #
    return (input_ids, input_mask, segment_ids, label_id)
    #
#
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
#
def convert_list_examples_to_batch(list_examples):
    """
    """
    (list_input_ids, list_input_mask, list_segment_ids,
     list_label_ids) = list(zip(*list_examples))
    
    batch_dict = {}
    batch_dict["input_ids"] = list_input_ids
    batch_dict["input_mask"] = list_input_mask
    batch_dict["segment_ids"] = list_segment_ids
    batch_dict["label_id"] = list_label_ids
    #
    return batch_dict
    #
#

