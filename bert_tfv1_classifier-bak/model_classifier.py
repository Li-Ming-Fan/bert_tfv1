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
    from modeling import BertConfig, BertModel
    from tokenization import FullTokenizer
except:
    from .modeling import BertConfig, BertModel
    from .tokenization import FullTokenizer


#
class ModelForPrediction(object):
    """
    """    
    def __init__(self, settings):
        #
        self.settings = settings
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = False,
                                          allow_soft_placement = True)
        self.sess_config.gpu_options.allow_growth = True
        #
    #
    
    #
    def feed_dict_predict(self, input_batch):        
        feed_dict = {}
        for tensor_tag, tensor in self._inputs_pred.items():
            feed_dict[tensor] = input_batch[tensor_tag]
        return feed_dict
    #
    def predict(self, input_batch):
        feed_dict = self.feed_dict_predict(input_batch)
        result_dict = self._sess.run(self._outputs_pred, feed_dict = feed_dict)   
        return result_dict
    #
    def set_inputs_outputs_pred(self, inputs_dict, outputs_dict):
        """
        """
        self._inputs_pred = {}
        self._outputs_pred = {}
        #
        for tensor_tag, tensor_name in inputs_dict.items():
            tensor = self._graph.get_tensor_by_name(tensor_name)
            self._inputs_pred[tensor_tag] = tensor
        #
        for tensor_tag, tensor_name in outputs_dict.items():
            tensor = self._graph.get_tensor_by_name(tensor_name)
            self._outputs_pred[tensor_tag] = tensor
        #
    #

    #
    def create_model_graph_and_sess(self, model_graph_creator, settings):
        """
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            model_graph_creator(settings)
        #
        self._sess = tf.Session(graph=self._graph, config = self.sess_config)
        #
    #
    def get_model_graph_and_sess(self):
        #
        return self._graph, self._sess
        #
    #
    def load_vars_from_ckpt(self, ckpt_path):
        #
        if os.path.isdir(ckpt_path):
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            ckpt_file_path = ckpt.model_checkpoint_path
        else:
            ckpt_file_path = ckpt_path
        #
        with self._graph.as_default():
            assignment_map = get_assignment_map_samename(ckpt_file_path)
            tf.train.init_from_checkpoint(ckpt_file_path, assignment_map)
            #
            self._sess.run(tf.global_variables_initializer())
            #
    #

    #        
    def prepare_for_prediction_with_pb(self, pb_path, inputs_dict, outputs_dict):
        """ load pb for prediction
        """
        if not os.path.exists(pb_path):
            assert False, 'ERROR: %s NOT exists' % pb_path
        #
        self._graph = tf.Graph()
        with self._graph.as_default():
            with open(pb_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
                #
                print('Graph loaded for prediction')
                #
            #
        #
        self._sess = tf.Session(graph = self._graph, config = self.sess_config)
        #
        self.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        #
    #
    def prepare_for_prediction_with_ckpt(self, model_graph_creator, settings,
                                         ckpt_path, inputs_dict, outputs_dict):
        """ load ckpt for prediction
        """
        self.create_model_graph_and_sess(model_graph_creator, settings)
        self.load_vars_from_ckpt(ckpt_path)
        #
        self.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        #
    #
    def save_pb_file(self, list_pb_save_names, pb_filepath):
        """
        """
        constant_graph = graph_util.convert_variables_to_constants(
                self._sess, self._sess.graph_def,
                output_node_names = list_pb_save_names)
        with tf.gfile.GFile(pb_filepath, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #
        str_info = 'pb_file saved: %s' % pb_filepath
        print(str_info)
        #
    #
#
def get_assignment_map_samename(init_ckpt, list_vars=None):
    """
    """
    if list_vars is None:
        list_vars = tf.global_variables()
    #
    name_to_variable = collections.OrderedDict()
    for var in list_vars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
        #
        print("variable name: %s" % name)
        #
    
    #
    ckpt_vars = tf.train.list_variables(init_ckpt)
    # 
    assignment_map = collections.OrderedDict()
    for x in ckpt_vars:
        (name, var) = (x[0], x[1])
        #
        if name not in name_to_variable:
            continue
        #
        assignment_map[name] = name
        print("assigned_variable name: %s" % name)
        #
    #    
    return assignment_map
    #
#

#
def create_model(bert_config, is_training, 
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

            return (loss, per_example_loss, logits, probabilities)

        else:
            return probabilities
        #
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
def convert_single_example(tokenizer, text_a, text_b=None, label_id=None,
                           max_seq_length=128, logger=None):
    """
    """
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

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



#
if __name__ == '__main__':
    """
    """
    config_file = "./chinese_L-12_H-768_A-12/bert_config.json"
    vocab_file = "./chinese_L-12_H-768_A-12/vocab.txt"
    dir_ckpt = "./model_sim/"
    #
    num_labels = 3
    pb_filename = "model_sim_frozen.pb"
    #
    # model
    bert_config = BertConfig(vocab_size=100)
    bert_config = bert_config.from_json_file(config_file)
    #
    print(bert_config.to_json_string())
    #
    upper_settings = {}
    upper_settings["num_labels"] = num_labels
    upper_settings["class_weights"] = None
    #
    settings_all = {}
    settings_all["bert_config"] = bert_config
    settings_all["upper_settings"] = upper_settings
    #
    inputs_dict = {}
    inputs_dict["input_ids"] = "input_ids:0"
    inputs_dict["input_mask"] = "input_mask:0"
    inputs_dict["segment_ids"] = "segment_ids:0"
    #
    outputs_dict = {}
    outputs_dict["probabilities"] = "loss/Softmax:0"
    # outputs_dict["bias"] = "bert/pooler/dense/bias:0"
    #
    model_for_prediction = ModelForPrediction({})
    model_for_prediction.prepare_for_prediction_with_ckpt(model_creator,
        settings_all, dir_ckpt, inputs_dict, outputs_dict)
    #
    model_for_prediction.save_pb_file([ "loss/Softmax" ],
        os.path.join(dir_ckpt, pb_filename) )
    #

    #
    model_for_prediction_new = ModelForPrediction({})
    model_for_prediction_new.prepare_for_prediction_with_pb(
        os.path.join(dir_ckpt, pb_filename), inputs_dict, outputs_dict)
    #

    #
    # tokenizer
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    #

    #
    # predict
    #
    text_a = "是"
    text_b = "是"
    #
    list_examples = [ convert_single_example(tokenizer, text_a, text_b,
                                             max_seq_length=256) ]
    batch_dict = convert_list_examples_to_batch(list_examples)
    #
    result_dict = model_for_prediction.predict(batch_dict)
    print(result_dict)
    #
    result_dict = model_for_prediction_new.predict(batch_dict)
    print(result_dict)
    #
