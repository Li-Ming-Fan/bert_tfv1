
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import argparse
import collections

import tensorflow as tf


#
from models.model_wrapper import ModelForTrain, ModelForPrediction
from models.model_wrapper import adamw_linear_schedule_creator
from models.model_wrapper import train_ops_creator_simple
from models.model_wrapper import train_ops_creator_multibatch
from models.settings_baseboard import SettingsBaseboard
#
from metrics_np import Metrics
#

#
from models import tokenization
from models.modeling_bert import BertConfig
from models.model_classifier import model_graph_creator
#
from training_data_processor import SimProcessor
from training_data_processor import file_based_convert_examples_to_features
from training_data_processor import file_based_input_fn_builder
#



#
data_processors = {
    "sim": SimProcessor,
}

#
task_name = "sim"
data_dir = "../data_sim"
output_dir = "../model_sim"
#
bert_config_file = "../chinese_L-12_H-768_A-12/bert_config.json"
vocab_file = "../chinese_L-12_H-768_A-12/vocab.txt"
init_checkpoint = "../chinese_L-12_H-768_A-12/bert_model.ckpt"
#
max_seq_length = 20
save_checkpoints_steps = 10
#
with_multibatch = 0
grad_accum_steps = 4
train_batch_size = 32
#
do_train = 1
do_eval = 1
do_predict = 0
#
gpu_id = "0"
#


#
def parsed_args():
    """
    """
    # Hyper Parameters
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task_name', default=task_name, type=str)
    parser.add_argument('--data_dir', default=data_dir, type=str)
    parser.add_argument('--output_dir', default=output_dir, type=str)
    #
    parser.add_argument('--vocab_file', default=vocab_file, type=str)
    parser.add_argument('--bert_config_file', default=bert_config_file, type=str)
    parser.add_argument('--init_checkpoint', default=init_checkpoint, type=str)
    #
    parser.add_argument('--do_lower_case', default=1, type=int)                   # bool
    parser.add_argument('--max_seq_length', default=max_seq_length, type=int)
    #
    parser.add_argument('--with_multibatch', default=with_multibatch, type=int)    # bool
    parser.add_argument('--grad_accum_steps', default=grad_accum_steps, type=int)
    #
    parser.add_argument('--gpu_id', default=gpu_id, type=str)
    #
    parser.add_argument('--do_train', default=do_train, type=int)       # bool
    parser.add_argument('--do_eval', default=do_eval, type=int)         # bool
    parser.add_argument('--do_predict', default=do_predict, type=int)   # bool
    #
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--train_batch_size', default=train_batch_size, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--predict_batch_size', default=8, type=int)
    #
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--save_checkpoints_steps', default=save_checkpoints_steps, type=int)
    #
    args = parser.parse_args()
    #
    return args
    #
#
def main(args):
    """
    """
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #
    settings = SettingsBaseboard()
    settings.assign_info_from_namedspace(args)
    #

    #
    # bert settings
    #
    bert_config = BertConfig.from_json_file(settings.bert_config_file)
    #
    if settings.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (settings.max_seq_length, bert_config.max_position_embeddings))
    #
    tokenization.validate_case_matches_checkpoint(settings.do_lower_case,
                                                  settings.init_checkpoint)
    # 
    tokenizer = tokenization.FullTokenizer(
        vocab_file=settings.vocab_file, do_lower_case=settings.do_lower_case)
    #

    #
    # task settings
    #
    task_name = settings.task_name.lower()
    if task_name not in data_processors:
        raise ValueError("Task not found: %s" % (task_name))
    #
    processor = data_processors[task_name]()
    label_list = processor.get_labels()
    try:
        class_weights = processor.get_class_weights()
    except:
        class_weights = None
    #
    num_labels = len(label_list)
    #
    upper_settings = {}
    upper_settings["num_labels"] = num_labels
    upper_settings["class_weights"] = class_weights
    #
    tf.gfile.MakeDirs(settings.output_dir)
    #
    # log
    #
    log_path = "log_%s_%s.txt" % (settings.task_name, settings.str_datetime)
    logger = settings.create_logger(log_path)
    #

    #
    settings.bert_config = bert_config
    settings.upper_settings = upper_settings
    #
    # settings.model_dir = settings.output_dir
    # settings.model_name = settings.task_name
    #

    #
    # task
    #
    inputs_dict = {}
    inputs_dict["input_ids"] = "input_ids:0"
    inputs_dict["input_mask"] = "input_mask:0"
    inputs_dict["segment_ids"] = "segment_ids:0"
    #
    outputs_dict = {}
    outputs_dict["probabilities"] = "loss/Softmax:0"
    #
    list_pb_save_names = [ "loss/Softmax" ]
    pb_filepath = os.path.join(settings.output_dir, "model_%s.pb" % settings.task_name)
    #

    #
    if settings.do_train:
        train_examples = processor.get_train_examples(settings.data_dir)
        num_train_steps = int(
            len(train_examples) / settings.train_batch_size * settings.num_train_epochs)
        num_warmup_steps = int(num_train_steps * settings.warmup_proportion)

        train_file = os.path.join(settings.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, settings.max_seq_length, tokenizer, train_file, logger)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", settings.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        #
        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            train_file, settings.max_seq_length, True, True)
        #
        if settings.with_multibatch:
            train_ops_creator = train_ops_creator_multibatch
            settings.batch_size = settings.train_batch_size // settings.grad_accum_steps
        else:
            train_ops_creator = train_ops_creator_simple
            settings.batch_size = settings.train_batch_size
        #
        settings.init_lr = settings.learning_rate
        settings.num_train_steps = num_train_steps
        settings.num_warmup_steps = num_warmup_steps
        #
        settings.max_train_steps = num_train_steps
        settings.model_dir = settings.output_dir
        settings.model_name = settings.task_name
        #

        #
        model = ModelForTrain(settings)
        model.dataset_creator = dataset_creator
        model.model_graph_creator = model_graph_creator
        model.optimizer_creator = adamw_linear_schedule_creator
        model.train_ops_creator = train_ops_creator
        #
        outputs = model.create_model_graph_and_sess()
        train_ops_dict = model.create_train_ops()
        #
        if settings.init_checkpoint and settings.init_checkpoint != "None":
            model.load_vars_from_ckpt(settings.init_checkpoint)
        #        
        model.train(settings)
        #
    #
    if settings.do_eval:
        eval_examples = processor.get_dev_examples(settings.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(settings.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, settings.max_seq_length, tokenizer, eval_file, logger)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", settings.eval_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            eval_file, settings.max_seq_length, False, False)
        #
        model = ModelForPrediction(settings)
        model.model_graph_creator = model_graph_creator
        #
        model.create_model_graph_and_sess()
        model.load_vars_from_ckpt(settings.output_dir)
        #
        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)
        #
        graph, sess = model.get_model_graph_and_sess()
        settings.batch_size = settings.eval_batch_size
        #
        list_labels = []
        list_pred_logits = []

        with graph.as_default():
            dataset = dataset_creator(settings)
            data_iter = dataset.make_one_shot_iterator()
            batch_data_tensor = data_iter.get_next()
            #
            try:
                while True:
                    batch_data = sess.run(batch_data_tensor)
                    output = model.predict(batch_data)
                    # print(batch_data)
                    # print(output)
                    #
                    labels = batch_data["label_ids"].tolist()
                    probs = output["probabilities"].tolist()
                    #
                    list_labels.extend(labels)
                    list_pred_logits.extend(probs)
                    #
            except:
                print("eval finished.")
            #

        #
        acc = Metrics.accuracy(list_pred_logits, list_labels)
        macro_f1 = Metrics.macro_f1(list_pred_logits, list_labels,
                list_classes=list(range(num_labels)))
        #
        scores = Metrics.classification_scores(list_pred_logits, list_labels,
                list_classes=list(range(num_labels)))
        #
        result = {}
        result["accuracy"] = acc
        result["macro_f1"] = macro_f1
        result["scores"] = scores
        #
        output_eval_file = os.path.join(settings.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #

    if settings.do_predict:
        predict_examples = processor.get_test_examples(settings.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(settings.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                settings.max_seq_length, tokenizer,
                                                predict_file, logger)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", settings.predict_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            predict_file, settings.max_seq_length, False, False)
        #
        model = ModelForPrediction(settings)
        mdoel.model_graph_creator = model_graph_creator
        #
        model.create_model_graph_and_sess()
        model.load_vars_from_ckpt(settings.output_dir)
        #
        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)
        #
        graph, sess = model.get_model_graph_and_sess()
        settings.batch_size = settings.eval_batch_size
        #
        list_labels = []
        list_pred_logits = []
        #
        list_batches_with_pred = []
        #
        with graph.as_default():
            dataset = dataset_creator(settings)
            data_iter = dataset.make_one_shot_iterator()
            batch_data_tensor = data_iter.get_next()
            #
            try:
                while True:
                    batch_data = sess.run(batch_data_tensor)
                    output = model.predict(batch_data)
                    # print(batch_data)
                    # print(output)
                    #
                    # labels = batch_data["label_ids"].tolist()
                    probs = output["probabilities"]
                    #
                    batch_data["pred_probs"] = probs
                    #
                    list_batches_with_pred.append(batch_data)
                    #
            except:
                print("prediction finished.")
            #
        #
        output_file = os.path.join(settings.output_dir, "pred_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for item in list_batches_with_pred:
                pass
            #
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #
    #
#

#
if __name__ == "__main__":
    """
    """
    args = parsed_args()
    main(args)
    #
#



