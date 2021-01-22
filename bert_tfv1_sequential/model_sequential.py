#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


try:
    from modeling import BertConfig, BertModel
    from tokenization import FullTokenizer
except:
    from .modeling import BertConfig, BertModel
    from .tokenization import FullTokenizer


#
class ModelBilstmCRF(object):
    """
    """
    def __init__(self, num_layers, hidden_unit, dropout_rate,
                 num_labels, is_training, cell_type="lstm"):
        """
        """
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.is_training = is_training

    #
    def forward(self, seq_embeded, seq_len, labels=None):
        """
        """
        if self.is_training:
            seq_embeded = tf.nn.dropout(seq_embeded, rate=self.dropout_rate)

        rnn_output = self.bilstm_layer_forward(seq_embeded, seq_len)
        logits = self.projection_forward(rnn_output)

        with tf.variable_scope('crf_layer'): 
            crf_trans = tf.get_variable("transitions",
                    shape=[self.num_labels, self.num_labels],
                    initializer=tf.constant_initializer(1.0 / self.num_labels) )
            #
            pred_ids, viterbi_score = crf.crf_decode(potentials=logits,
                sequence_length=seq_len, transition_params=crf_trans)
            #

        if self.is_training:
            loss, crf_trans = self.get_crf_loss(logits, labels, seq_len, crf_trans)
            return (loss, logits, pred_ids, crf_trans)
        else:
            return (viterbi_score, logits, pred_ids, crf_trans)
    #
    def _get_rnn_cell(self):
        """
        """
        rnn_cell = None
        if self.cell_type == 'lstm':
            rnn_cell = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            rnn_cell = rnn.GRUCell(self.hidden_unit)

        if self.is_training and self.dropout_rate is not None:
            keep_prob = 1 - self.dropout_rate
            rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)

        return rnn_cell

    def bilstm_layer_forward(self, inputs_embedded, seq_len):
        """
        """
        if self.num_layers == 0:
            return inputs_embedded
        #
        outputs = inputs_embedded
        #
        with tf.variable_scope('rnn_layers'):
            for idx in range(self.num_layers):
                with tf.variable_scope('%d' % idx):
                    cell_fw = self._get_rnn_cell()
                    cell_bw = self._get_rnn_cell()
                    #
                    outputs_rnn, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, outputs, seq_len, dtype=tf.float32)
                    # [B, T, D]
                    outputs = tf.concat(outputs_rnn, axis=2)
                    #
                    outputs = tf.layers.dense(outputs, self.hidden_unit * 2)
                    outputs = tf.nn.tanh(outputs)
                    #
            #
        #
        return outputs
        #

    def projection_forward(self, rnn_outputs, name=None):
        """
        """
        with tf.variable_scope("project" if not name else name):
            seq_d = tf.layers.dense(rnn_outputs, self.hidden_unit,
                                    activation=tf.tanh, name="intermediate")
            logits = tf.layers.dense(seq_d, self.num_labels, name="logits")
            return logits

    def get_crf_loss(self, logits, labels, seq_len, crf_trans):
        """
        """
        with tf.variable_scope("crf_loss"):
            log_likelihood, crf_trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits, tag_indices=labels, sequence_lengths=seq_len,
                transition_params = crf_trans )
            crf_loss = - tf.reduce_mean(log_likelihood)
        #
        return crf_loss, crf_trans
        #
    #
#
def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, label_ids, upper_settings):
    """
    """
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )
    #
    seq_embeded = model.get_sequence_output()
    #

    #
    num_layers = upper_settings["num_layers"]
    hidden_unit = upper_settings["hidden_unit"]
    dropout_rate = upper_settings["dropout_rate"]
    num_labels = upper_settings["num_labels"]
    cell_type = upper_settings["cell_type"]
    only_text_a = upper_settings["only_text_a"]
    #
    # seq_mask = tf.sign(tf.abs(input_ids))  # [B, T]
    #
    if only_text_a:
        seq_len_a = tf.reduce_sum(input_mask, reduction_indices=1)  # [B, ]
    else:
        seq_len_all = tf.reduce_sum(input_mask, reduction_indices=1)  # [B, ]
        seq_len_b = tf.reduce_sum(segment_ids, reduction_indices=1)  # [B, ]
        seq_len_a = seq_len_all - seq_len_b
    #

    bilstm_crf = ModelBilstmCRF(num_layers, hidden_unit, dropout_rate, num_labels,
                        is_training=is_training, cell_type=cell_type)
    outputs = bilstm_crf.forward(seq_embeded, seq_len_a, label_ids)

    return outputs
#

#
def model_creator(settings, input_tensors=None):
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
    bert_config = settings["bert_config"]
    upper_settings = settings["upper_settings"]
    #
    outputs = create_model(bert_config, is_training, 
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
def convert_single_example(tokenizer, text_a, text_b=None, idx_labels=None,
                           max_seq_length=128):
    """
    """
    # text_a = convert_quan_to_ban(text_a)
    tokens_a = tokenizer.tokenize(text_a)
    #
    tokens_b = None
    if text_b:
        # text_b = convert_quan_to_ban(text_b)
        tokens_b = tokenizer.tokenize(text_b)
    #
    example_ids = convert_single_example_tokens(
        tokenizer, tokens_a, tokens_b, idx_labels, max_seq_length)    
    #
    return example_ids
    #
#
def convert_single_example_tokens(tokenizer, tokens_a, tokens_b=None, idx_labels=None,
                                  max_seq_length=128):
    """
    """
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        max_len_a = max_seq_length - 3 - len(tokens_b)
        if len(tokens_a) > max_len_a:
            tokens_a = tokens_a[0:max_len_a]
            if idx_labels:
                idx_labels = idx_labels[0:max_len_a]
    else:
        # Account for [CLS] and [SEP] with "- 2"
        max_len_a = max_seq_length - 2
        if len(tokens_a) > max_len_a:
            tokens_a = tokens_a[0:max_len_a]
            if idx_labels:
                idx_labels = idx_labels[0:max_len_a]

    tokens = []
    segment_ids = []
    label_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)
    #
    tokens.extend(tokens_a)
    segment_ids.extend( [0] * len(tokens_a))
    if idx_labels:
        label_ids.extend(idx_labels)
    else:
        label_ids.extend( [0] * len(tokens_a))
    #
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    if tokens_b:
        tokens.extend(tokens_b)
        segment_ids.extend( [1] * len(tokens_b))
        label_ids.extend([0] * len(tokens_b))
        #
        tokens.append("[SEP]")
        segment_ids.append(1)
        label_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    d = max_seq_length - len(input_ids)
    if d > 0:
        input_ids.extend( [0] *d )
        input_mask.extend( [0] *d )
        segment_ids.extend( [0] *d )
        label_ids.extend( [0] *d )

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    #
    return (input_ids, input_mask, segment_ids, label_ids)
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
    batch_dict["label_ids"] = list_label_ids
    #
    return batch_dict
    #
#
def convert_quan_to_ban(str_quan):
    """全角转半角"""
    str_ban = ""
    for uchar in str_quan:
        inside_code = ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        #
        str_ban += chr(inside_code)
    return str_ban
#



#
if __name__ == '__main__':
    """
    """
    pass

