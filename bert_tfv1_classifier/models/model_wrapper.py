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

#
class ModelForTrain(object):
    """ dataset = dataset_creator(settings)
        outputs = model_graph_creator(settings, batch_data_tensors=None)
        optimizer = optimizer_creator(loss_tensor, settings)
        train_ops_creator = train_ops_creator(loss_tensor, optimizer, settings)
    """
    def __init__(self, settings):
        #
        # graph-related
        #
        self.dataset_creator = None  # dataset_creator(settings)
        self.model_graph_creator = None  # model_graph_creator(settings, batch_data_tensors)
        self.optimizer_creator = None    # optimizer_creator(loss_tensor, settings)
        self.train_ops_creator = None    # train_ops_creator(loss_tensor, optimizer, settings)
        #
        # settings
        # 
        self.settings = settings
        #
        if "logger" not in settings.__dict__:
            self.logger = create_logger(settings.log_path, settings.log_path)
        else:
            self.logger = settings.logger
        #
        info_dict = {}
        for name, value in self.settings.__dict__.items():
            if not isinstance(value, (int, float, str, bool, list, dict, tuple)):
                continue
            info_dict[str(name)] = value        
        #
        self.logger.info("settings: %s" % str(info_dict))
        #
        # session info
        self.sess_config = tf.ConfigProto(log_device_placement = False,
                                          allow_soft_placement = True)
        self.sess_config.gpu_options.allow_growth = True
        #
    #
    def print_and_log(self, info):
        """
        """
        print(info)
        self.logger.info(info)
        #
    #

    #
    def create_model_graph_and_sess(self):
        """ returning: model_outputs
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._dataset = self.dataset_creator(self.settings)
            self._data_iter = self._dataset.make_one_shot_iterator()
            self._batch_data_tensors = self._data_iter.get_next()
            #
            outputs = self.model_graph_creator(self.settings, self._batch_data_tensors)
            self._loss_tensor_train = outputs["loss_train"]
            #
        #
        self._sess = tf.Session(graph=self._graph, config = self.sess_config)
        #
        # self._model_outputs = outputs
        return outputs
        #
    #
    def create_train_ops(self):
        """ returning: train_ops_dict
        """
        with self._graph.as_default():
            optimizer = self.optimizer_creator(self._loss_tensor_train, self.settings)
            train_ops_dict = self.train_ops_creator(self._loss_tensor_train,
                optimizer, self.settings)
            #
            self._train_ops_dict = train_ops_dict
            #
        #
        return train_ops_dict
        #
    #
    def remove_from_trainable_variables(self, non_trainable_names, from_all=False):
        """
        """
        with self._graph.as_default():
            if from_all:
                tr_var = tf.global_variables()
            else:
                tr_var = None
            #
            remove_from_trainable_variables(non_trainable_names, tr_var)
            #
        #
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
    def train(self, settings):
        """ train(self, settings):
            comment the lines: self._print_debug_outputs(outputs)

            debug(self, settings):
            uncomment the lines: self._print_debug_outputs(outputs)
        """
        if "accum_ops" in self._train_ops_dict:
            assert "grad_accum_steps" in settings.__dict__, "grad_accum_steps not in settings"
            grad_accum_steps = settings.grad_accum_steps
            #
            accum_ops = self._train_ops_dict["accum_ops"]
            zero_ops = self._train_ops_dict["zero_ops"]
            #
            train_op = self._train_ops_dict["train_op"]
            #
            train_ops_dict_filtered = {}
            for key, value in self._train_ops_dict.items():
                if key not in ["accum_ops", "zero_ops"]:
                    train_ops_dict_filtered[key] = value
            #
        else:
            accum_ops = None
            train_ops_dict_filtered = self._train_ops_dict
        #

        #
        sess = self._sess
        with self._graph.as_default():
            #
            save_checkpoints_steps = settings.save_checkpoints_steps
            max_train_steps = settings.max_train_steps
            model_dir = settings.model_dir
            model_name = settings.model_name
            #
            self._saver = tf.train.Saver()
            #
            sess.run(tf.global_variables_initializer())
            loss = 10.0
            #
            if accum_ops:
                for step in range(max_train_steps):
                    #
                    if step % save_checkpoints_steps == 0:
                        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                                         global_step = step)
                        self.logger.info("step: %d, loss: %f" % (step, loss))
                    #
                    sess.run(zero_ops)
                    for grad_step in range(grad_accum_steps):
                        accum_array = sess.run(accum_ops)
                    #
                    outputs = sess.run(train_ops_dict_filtered)
                    loss = outputs["loss_train"]
                    #
                    # debug
                    # self._print_debug_outputs(outputs)
                    #
            else:
                for step in range(max_train_steps):
                    #
                    if step % save_checkpoints_steps == 0:
                        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                                         global_step = step)
                        self.logger.info("step: %d, loss: %f" % (step, loss))
                    #
                    # _, loss_value = sess.run([train_op, loss])
                    outputs = sess.run(train_ops_dict_filtered)
                    loss = outputs["loss_train"]
                    #
                    # debug
                    # self._print_debug_outputs(outputs)
                    #
            #
            self._saver.save(self._sess, os.path.join(model_dir, model_name),
                             global_step = max_train_steps)
            #
        #
    #
    def debug(self, settings):
        """ train(self, settings):
            comment the lines: self._print_debug_outputs(outputs)

            debug(self, settings):
            uncomment the lines: self._print_debug_outputs(outputs)
        """
        if "accum_ops" in self._train_ops_dict:
            assert "grad_accum_steps" in settings.__dict__, "grad_accum_steps not in settings"
            grad_accum_steps = settings.grad_accum_steps
            #
            accum_ops = self._train_ops_dict["accum_ops"]
            zero_ops = self._train_ops_dict["zero_ops"]
            #
            train_op = self._train_ops_dict["train_op"]
            #
            train_ops_dict_filtered = {}
            for key, value in self._train_ops_dict.items():
                if key not in ["accum_ops", "zero_ops"]:
                    train_ops_dict_filtered[key] = value
            #
        else:
            accum_ops = None
            train_ops_dict_filtered = self._train_ops_dict
        #

        #
        sess = self._sess
        with self._graph.as_default():
            #
            save_checkpoints_steps = settings.save_checkpoints_steps
            max_train_steps = settings.max_train_steps
            model_dir = settings.model_dir
            model_name = settings.model_name
            #
            self._saver = tf.train.Saver()
            #
            sess.run(tf.global_variables_initializer())
            loss = 10.0
            #
            if accum_ops:
                for step in range(max_train_steps):
                    #
                    if step % save_checkpoints_steps == 0:
                        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                                         global_step = step)
                        self.logger.info("step: %d, loss: %f" % (step, loss))
                    #
                    sess.run(zero_ops)
                    for grad_step in range(grad_accum_steps):
                        accum_array = sess.run(accum_ops)
                    #
                    outputs = sess.run(train_ops_dict_filtered)
                    loss = outputs["loss_train"]
                    #
                    # debug
                    self._print_debug_outputs(outputs)
                    #
            else:
                for step in range(max_train_steps):
                    #
                    if step % save_checkpoints_steps == 0:
                        self._saver.save(self._sess, os.path.join(model_dir, model_name),
                                         global_step = step)
                        self.logger.info("step: %d, loss: %f" % (step, loss))
                    #
                    # _, loss_value = sess.run([train_op, loss])
                    outputs = sess.run(train_ops_dict_filtered)
                    loss = outputs["loss_train"]
                    #
                    # debug
                    self._print_debug_outputs(outputs)
                    #
            #
            self._saver.save(self._sess, os.path.join(model_dir, model_name),
                             global_step = max_train_steps)
            #
        #
    #
    def _print_debug_outputs(self, outputs_dict):
        """
        """
        for tensor_tag, tensor_value in outputs_dict.items():
            if tensor_tag.startswith("loss") or tensor_tag.startswith("debug"):
                self.print_and_log("-" * 60)
                self.print_and_log("tensor_tag: %s" % tensor_tag)
                self.print_and_log(tensor_value)
        #
        self.print_and_log("-" * 60)
        #
    #

    #
    def print_all_variables(self):
        """
        """
        with self._graph.as_default():
            global_vars = tf.global_variables()
            local_vars = tf.local_variables()
            #
            self.print_and_log("global_variables:")
            for item in global_vars:
                self.print_and_log(item)
            self.print_and_log("global_variables end.")
            #
            self.print_and_log("local_variables:")
            for item in local_vars:
                self.print_and_log(item)
            self.print_and_log("local_variables end.")
            #
    #
    def print_trainable_variables(self):
        """
        """
        with self._graph.as_default():
            tr_vars = tf.trainable_variables()
            #
            self.print_and_log("trainable_variables:")
            for item in tr_vars:
                self.print_and_log(item)
            self.print_and_log("trainable_variables end.")
            #
    #
    def get_tensors_dict(self, tensor_names_dict):
        """
        """
        tensors_dict = {}
        for tensor_tag, tensor_name in tensor_names_dict.items():
            tensor = self._graph.get_tensor_by_name(tensor_name)
            tensors_dict[tensor_tag] = tensor
        #
        return tensors_dict
        #
    #
#
class ModelForPrediction(object):
    """
    """    
    def __init__(self, settings):
        #
        self.model_graph_creator = None
        #
        # settings
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
    def create_model_graph_and_sess(self):
        """
        """
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.model_graph_creator(self.settings)
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
    def prepare_for_prediction_with_ckpt(self, ckpt_path, inputs_dict, outputs_dict):
        """ load ckpt for prediction
        """
        self.create_model_graph_and_sess()
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
def remove_from_trainable_variables(non_trainable_names, trainable_vars=None):
    """
    """
    graph = tf.get_default_graph()
    #
    if trainable_vars is None:
        trainable_vars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # tf.trainable_variables()
        
    #    
    graph.clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #
    for var in trainable_vars:
        for item in non_trainable_names:
            if item in var.name:
                logger = get_logger(log_tag)
                logger.info("not_training: %s" % var.name)
                break
        else:
            # print("training: %s" % var.name)
            graph.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        #
    #
#
def train_ops_creator_simple(loss_tensor, optimizer, settings):
    """
    """
    if "clip_norm" in settings.__dict__:
        clip_norm = settings.clip_norm
    else:
        clip_norm = 1.0
    #

    #
    tr_vars = tf.trainable_variables()
    global_step = tf.train.get_or_create_global_step()
    #
    grads = tf.gradients(loss_tensor, tr_vars)
    #
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    #
    train_op = optimizer.apply_gradients(
        zip(grads, tr_vars), global_step=global_step)
    #
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    #
    train_related = {}
    train_related["train_op"] = train_op
    train_related["loss_train"] = loss_tensor
    #
    return train_related
    #
#
def train_ops_creator_multibatch(loss_tensor, optimizer, settings):
    """
    """
    if "clip_norm" in settings.__dict__:
        clip_norm = settings.clip_norm
    else:
        clip_norm = 1.0
    #

    #
    tr_vars = tf.trainable_variables()
    global_step = tf.train.get_or_create_global_step()
    #
    grads = tf.gradients(loss_tensor, tr_vars)
    #
    accum_vars = [ tf.Variable(tf.zeros_like(
        v.initialized_value()), trainable=False) for v in tr_vars ]
    #
    zero_ops = [ v.assign(tf.zeros_like(v)) for v in accum_vars ]
    #
    accum_ops = []
    for i, g in enumerate(grads):
        # print(i)
        # print(g)
        if g is not None:
            accum_ops.append( accum_vars[i].assign_add(g) )
        #
    #
    # accum_ops = [ accum_vars[i].assign_add(g) for i, g in enumerate(grads)]
    #
    (grads, _) = tf.clip_by_global_norm(accum_vars, clip_norm=clip_norm)
    #
    train_op = optimizer.apply_gradients(
        zip(grads, tr_vars), global_step=global_step)
    #
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    #
    train_related = {}
    train_related["train_op"] = train_op
    train_related["zero_ops"] = zero_ops
    train_related["accum_ops"] = accum_ops
    train_related["loss_train"] = loss_tensor
    #
    return train_related
    #
#

#
def adamw_constant_schedule_creator(loss, settings):
    """
    """
    init_lr = settings.init_lr
    #
    if "exclude_from_weight_decay" in settings.__dict__:
        exclude_from_weight_decay = settings.exclude_from_weight_decay
    else:
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    #

    #
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # global_step = tf.train.get_or_create_global_step()

    #
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=exclude_from_weight_decay)
        # exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    #
    return optimizer
    #
#
def adamw_linear_schedule_creator(loss, settings):
    """
    """
    init_lr = settings.init_lr
    num_train_steps = settings.num_train_steps
    num_warmup_steps = settings.num_warmup_steps
    #
    if "exclude_from_weight_decay" in settings.__dict__:
        exclude_from_weight_decay = settings.exclude_from_weight_decay
    else:
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
    #

    #
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    global_step = tf.train.get_or_create_global_step()
    
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(learning_rate, 
        global_step, num_train_steps, end_learning_rate=0.0, power=1.0, cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=exclude_from_weight_decay)
        # exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    #
    return optimizer
    #
#
class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """ A basic Adam optimizer that includes "correct" L2 weight decay.
    """
    def __init__(self, learning_rate, weight_decay_rate=0.0,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-6, exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """ Constructs a AdamWeightDecayOptimizer.
        """
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """ See base class.
        """
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                            tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                m.assign(next_m),
                v.assign(next_v)])
            #
        #
        return tf.group(*assignments, name=name)
        #

    def _do_use_weight_decay(self, param_name):
        """ Whether to use L2 weight decay for `param_name`.
        """
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """ Get the variable name from the tensor name.
        """
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
#

#
def create_logger(log_tag, log_path):
    """
    """
    logger = logging.getLogger(log_tag)  # use log_tag as log_name
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding='utf-8') 
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # self.logger.info('test')
    #
    return logger
    #
#
def get_logger(log_tag):
    """
    """
    logger = logging.getLogger(log_tag)  # use log_tag as log_name
    return logger
#

#
if __name__ == '__main__':
    """
    """
    pass
