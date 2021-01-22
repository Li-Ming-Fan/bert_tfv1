
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import collections

import json
import codecs
import csv
import pandas as pd

import tensorflow as tf

import tokenization
from modeling import BertConfig

from model_wrapper import create_logger, get_logger
from model_wrapper import ModelForTrain, ModelForPrediction, create_optimizer
from model_wrapper import train_op_dict_creator_simple
from model_wrapper import train_op_dict_creator_multibatch
from metrics_np import Metrics

from model_sequential import model_creator
from model_sequential import convert_quan_to_ban, convert_single_example_tokens
from data_mrc_seq import load_data_bio

#
task_name = "excavation"
data_dir = "../data_excavation"
output_dir = "../model_excavation"
#
bert_config_file = "../chinese_L-12_H-768_A-12/bert_config.json"
vocab_file = "../chinese_L-12_H-768_A-12/vocab.txt"
init_checkpoint = "../chinese_L-12_H-768_A-12/bert_model.ckpt"
#
max_seq_length = 20
save_checkpoints_steps = 10
#
with_multibatch = 0
grad_accum_steps = 2
train_batch_size = 32
#
do_train = True
do_eval = True
do_predict = False
do_debug= False
#
gpu_id = "0"
#
num_layers = 2
hidden_unit = 256
dropout_rate = 0.3
cell_type = "lstm"
grad_clip= 5.0
#


#
flags = tf.flags
FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string("task_name", task_name,
    "The name of the task to train.")

flags.DEFINE_string("data_dir", data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string("output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_string("bert_config_file", bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", vocab_file,
                    "The vocabulary file that the BERT model was trained on.")


## Other parameters

flags.DEFINE_string("init_checkpoint", init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

#
flags.DEFINE_bool("with_multibatch", with_multibatch,
    "Whether to use multibatch.")

flags.DEFINE_integer("grad_accum_steps", grad_accum_steps,
    "grad_accum_steps.")

#
flags.DEFINE_bool("do_train", do_train,
    "Whether to run training.")

flags.DEFINE_bool("do_eval", do_eval,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", do_predict,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_debug", do_debug,
    "Whether to run debug.")

flags.DEFINE_string("gpu_id", gpu_id, "gpu_id")

#
flags.DEFINE_integer("train_batch_size", train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 2,
                   "Total number of training epochs to perform.")

#
flags.DEFINE_float("warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", save_checkpoints_steps,
                     "How often to save the model checkpoint.")

#
flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')
flags.DEFINE_integer('num_layers', num_layers, 'number of rnn layers, default is 1')
flags.DEFINE_integer('hidden_unit', hidden_unit, 'size of lstm units')
flags.DEFINE_string('cell_type', cell_type, 'which rnn cell used')
flags.DEFINE_float('dropout_rate', dropout_rate, 'Dropout rate')
flags.DEFINE_float('grad_clip', grad_clip, 'Gradient clip')


#
class InputExample(object):
    """
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
#
class InputFeatures(object):
    """
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        """
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
#
def convert_single_example(ex_index, example, labels_str_to_idx, max_seq_length,
                           tokenizer, logger=None):
    """
    """
    tokens_a = example.text_a.split(' ')
    labels_str_list = example.label.split(' ')
    #
    # tokens_a = []
    # for token in tokens_a_raw:
    #     # token = convert_quan_to_ban(token)
    #     token_seg = tokenizer.tokenize(token)
    #     tokens_a.append( "".join(token_seg) )
    #
    assert len(tokens_a) == len(labels_str_list), "Error, len(tokens_a) != len(labels_str_list)"
    #
    if example.text_b:
        tokens_b = example.text_b.split(' ')
        # tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = None
    #
    idx_labels = [labels_str_to_idx.get(item, 0) for item in labels_str_list]
    #

    #
    input_ids, input_mask, segment_ids, label_ids = convert_single_example_tokens(
        tokenizer, tokens_a, tokens_b, idx_labels, max_seq_length)

    #
    if ex_index < 5 and logger:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens_a: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_a]))
        if tokens_b:
            logger.info("tokens_b: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_b]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    #
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    #
    return feature
    #
#
def file_based_convert_examples_to_features(
        examples, labels_str_to_idx, max_seq_length, tokenizer, output_file, logger=None):
    """
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    #
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0 and logger:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, labels_str_to_idx,
                                         max_seq_length, tokenizer, logger)
        
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        #
    #
    writer.close()
    #
#
def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    #
    return input_fn
    #
#

#
class DataProcessor(object):
    """
    """
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    set_labels_all = set()

    @classmethod
    def _read_data(cls, input_file):
        """Read BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    word = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[-1]
                    DataProcessor.set_labels_all.add(label)
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                words.append(word)
                labels.append(label)
            return lines
#
class NerProcessor(DataProcessor):
    """
    """
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        # list_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        list_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        return list_labels

    def get_positives(self):
        list_posi = [1, 2, 3, 4, 5, 6]
        return list_posi

    def has_only_text_a(self):
        return 1

    def _create_example(self, lines, dataset_type):
        """
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            #
            # text = line[1].strip()
            # label = line[0].strip()
            #
            text = convert_quan_to_ban(text)
            # print(text)
            #
            if i == 0:
                print(text)
                print(label)
            #
            examples.append(InputExample(
                guid=guid, text_a=text, text_b=text[0:5], label=label))
            #
        #
        return examples
        #
#
class ExcavationProcessor(DataProcessor):
    """
    """
    def get_train_examples(self, data_dir):
        return self._create_example_from_file(
            os.path.join(data_dir, "examples_train.txt"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example_from_file(
            os.path.join(data_dir, "examples_valid.txt"), "valid")

    def get_test_examples(self, data_dir):
        return self._create_example_from_file(
            os.path.join(data_dir, "examples_valid.txt"), "test")

    def get_labels(self):
        # list_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        list_labels = ["O", "B-ENT", "I-ENT", "B-DIR", "I-DIR"]
        return list_labels

    def get_positives(self):
        list_posi = [1, 2, 3, 4]
        return list_posi

    def has_only_text_a(self):
        return 0

    def _create_example_from_file(self, file_path, set_type):
        """
        """
        data_bio = load_data_bio(file_path)

        examples = []
        for (i, item) in enumerate(data_bio):
            guid = "%s-%s" % (set_type, i)
            #
            tokens_a = item["tokens_a"]
            tokens_b = item["tokens_b"]
            labels = item["labels"]
            #
            if len(tokens_a) != len(labels):
                print("len(tokens_a) != len(labels)")
                print(item)
                continue
            #
            text_a = " ".join(tokens_a)
            text_b = " ".join(tokens_b)
            label_str = " ".join(labels)
            #
            if i == 0:
                print(text_a)
                print(text_b)
                print(label_str)
            #
            ##
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label_str) )
            #
        #
        return examples
        #
#


#
def main(args):
    """
    """
    str_datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
    log_path = "log_%s_%s.txt" % (FLAGS.task_name, str_datetime)
    log_tag = log_path
    logger = create_logger(log_tag, log_path)

    settings_all = {}
    settings_all["logger"] = logger

    #
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

    processors = {
        "excavation": ExcavationProcessor,
    }

    #
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    list_labels = processor.get_labels()
    labels_str_to_idx = {}
    labels_idx_to_str = {}
    for idx, label in enumerate(list_labels):
        labels_str_to_idx[label] = idx
        labels_idx_to_str[idx] = label

    list_positives = processor.get_positives()
    only_text_a = processor.has_only_text_a()

    num_labels = len(list_labels)

    upper_settings = {}
    upper_settings["num_layers"] = FLAGS.num_layers
    upper_settings["hidden_unit"] = FLAGS.hidden_unit
    upper_settings["dropout_rate"] = FLAGS.dropout_rate
    upper_settings["cell_type"] = FLAGS.cell_type
    upper_settings["num_labels"] = num_labels   #
    upper_settings["list_positives"] = list_positives
    upper_settings["only_text_a"] = only_text_a

    file_path = "model_sequential_upper_settings.json"
    #
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(upper_settings, fp, ensure_ascii=False, indent=4)

    #
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    #
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    #
    settings_all["bert_config"] = bert_config
    settings_all["upper_settings"] = upper_settings
    settings_all["model_dir"] = FLAGS.output_dir
    settings_all["model_name"] = FLAGS.task_name

    #
    inputs_dict = {}
    inputs_dict["input_ids"] = "input_ids:0"
    inputs_dict["input_mask"] = "input_mask:0"
    inputs_dict["segment_ids"] = "segment_ids:0"
    #
    outputs_dict = {}
    outputs_dict["seq_preds"] = "crf_layer/cond/Merge:0"
    # outputs_dict["seq_preds"] = "crf_layer/ReverseSequence_1:0"
    #
    list_pb_save_names = [ "crf_layer/cond/Merge" ]
    pb_filepath = os.path.join(FLAGS.output_dir, "model_%s.pb" % FLAGS.task_name)
    #

    #
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, labels_str_to_idx, FLAGS.max_seq_length, tokenizer, train_file, logger)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        #
        if FLAGS.with_multibatch:
            train_op_dict_creator = train_op_dict_creator_multibatch
            settings_all["grad_accum_steps"] = FLAGS.grad_accum_steps
            settings_all["batch_size"] = FLAGS.train_batch_size // FLAGS.grad_accum_steps
        else:
            train_op_dict_creator = train_op_dict_creator_simple
            settings_all["batch_size"] = FLAGS.train_batch_size
        #
        settings_all["max_train_steps"] = num_train_steps
        settings_all["save_checkpoints_steps"] = FLAGS.save_checkpoints_steps
        #

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            train_file, FLAGS.max_seq_length, True, True)

        model = ModelForTrain(settings_all)
        outputs = model.create_model_graph_and_sess(
            dataset_creator, model_creator, settings_all)

        if FLAGS.init_checkpoint and FLAGS.init_checkpoint != "None":
            model.load_vars_from_ckpt(FLAGS.init_checkpoint)

        loss_train = outputs[0]

        def optimizer_creator(loss_tensor, settings):
            learning_rate=FLAGS.learning_rate,
            opt = create_optimizer(loss_tensor, init_lr=learning_rate,
                num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
            return opt

        train_op_dict = model.create_train_op_dict(optimizer_creator,
                train_op_dict_creator, loss_train, settings_all)

        if FLAGS.do_debug:
            # model.print_all_variables()
            #
            tensor_names_dict = {
                "loss_train_debug": "crf_layer/ReverseSequence_1:0",
            }
            #
            debug_tensors_dict = model.get_tensors_dict(tensor_names_dict)
            train_op_dict.update(debug_tensors_dict)
            #
            model.debug(train_op_dict, settings_all)
            #
        else:
            model.train(train_op_dict, settings_all)
            #

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, labels_str_to_idx, FLAGS.max_seq_length, tokenizer, eval_file, logger)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            eval_file, FLAGS.max_seq_length, False, False)

        model = ModelForPrediction(settings_all)
        model.create_model_graph_and_sess(model_creator, settings_all)
        model.load_vars_from_ckpt(FLAGS.output_dir)

        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)

        graph, sess = model.get_model_graph_and_sess()
        settings_all["batch_size"] = FLAGS.eval_batch_size

        list_labels_idx = []
        list_preds_idx = []

        with graph.as_default():
            dataset = dataset_creator(settings_all)
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
                    seq_preds = output["seq_preds"].tolist()
                    #
                    list_labels_idx.extend(labels)
                    list_preds_idx.extend(seq_preds)
                    #
            except:
                print("eval finished.")
            #
        #
        list_labels_str = []
        for seq in list_labels_idx:
            seq_str = [labels_idx_to_str[item] for item in seq]
            list_labels_str.append(seq_str)
        #
        list_preds_str = []
        for seq in list_preds_idx:
            seq_str = [labels_idx_to_str[item] for item in seq]
            list_preds_str.append(seq_str)
        #
        # print(len(list_labels_str))
        # print(len(list_preds_str))
        # print(list_labels_str[0])
        # print(list_preds_str[0])
        # print(labels_idx_to_str)
        #
        results = Metrics.seq_tagging_f1score(list_preds_str, list_labels_str)
        #
        result = {}
        result["f1_score"] = results[0]
        result["precision"] = results[1]
        result["recall"] = results[2]
        #
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, labels_str_to_idx,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, logger)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", FLAGS.predict_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            predict_file, FLAGS.max_seq_length, False, False)

        model = ModelForPrediction(settings_all)
        model.create_model_graph_and_sess(model_creator, settings_all)
        model.load_vars_from_ckpt(FLAGS.output_dir)

        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)

        graph, sess = model.get_model_graph_and_sess()
        settings_all["batch_size"] = FLAGS.eval_batch_size

        list_batches_with_pred = []

        list_labels_idx = []
        list_preds_idx = []

        with graph.as_default():
            dataset = dataset_creator(settings_all)
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
                    labels = batch_data["label_ids"]
                    seq_preds = output["seq_preds"]
                    #
                    list_labels_idx.extend(labels.tolist())
                    list_preds_idx.extend(seq_preds.tolist())
                    #
                    batch_data["seq_preds"] = seq_preds
                    list_batches_with_pred.append(batch_data)
                    #
            except:
                print("eval finished.")
            #
        #
        list_labels_str = []
        for seq in list_labels_idx:
            seq_str = [labels_idx_to_str[item] for item in seq]
            list_labels_str.append(seq_str)
        #
        list_preds_str = []
        for seq in list_preds_idx:
            seq_str = [labels_idx_to_str[item] for item in seq]
            list_preds_str.append(seq_str)
        #
        # print(len(list_labels_str))
        # print(len(list_preds_str))
        # print(list_labels_str[0])
        # print(list_preds_str[0])
        # print(labels_idx_to_str)
        #
        results = Metrics.seq_tagging_f1score(list_preds_str, list_labels_str)
        #
        result = {}
        result["f1_score"] = results[0]
        result["precision"] = results[1]
        result["recall"] = results[2]
        #

        #
        output_file = os.path.join(FLAGS.output_dir, "pred_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            #
            writer.write("{}\n".format(result))
            #
            for item in list_batches_with_pred:
                input_ids = item["input_ids"]
                input_mask = item["input_mask"]
                segment_ids = item["segment_ids"]
                label_ids = item["label_ids"]
                seq_preds = item["seq_preds"]
                #
                num_examples = len(input_ids)
                for eid in range(num_examples):
                    input_ids_curr = input_ids[eid]
                    input_mask_curr = input_mask[eid]
                    segment_ids_curr = segment_ids[eid]
                    label_ids_curr = label_ids[eid]
                    seq_preds_curr = seq_preds[eid]
                    #
                    tokens = tokenizer.convert_ids_to_tokens(input_ids_curr)
                    #
                    writer.write("%s\n" % ("=" % 60))
                    writer.write("{}\n".format(tokens))
                    writer.write("{}\n".format(label_ids_curr))
                    writer.write("{}\n".format(seq_preds_curr))
                    writer.write("\n")
                    #
            #
            writer.write("%s\n" % ("=" % 60))
            #
        #
#


if __name__ == "__main__":
    """
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    # flags.FLAGS.set_default('do_train', False)
    # flags.FLAGS.set_default('do_eval', False)
    # flags.FLAGS.set_default('do_predict', True)
    """
    tf.app.run()
